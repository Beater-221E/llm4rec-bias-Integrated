"""MiniOneRec workflow — Semantic ID generative retrieval (behavior-preserving)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.components.dataset.base import DatasetBundle
from llm4rec.components.dataset.movielens import build_movielens_bundle
from llm4rec.components.evaluation.generation import GenerationMetrics, evaluate_sid_checkpoint
from llm4rec.core.context import ExperimentContext
from llm4rec.core.exceptions import ConfigurationError
from llm4rec.core.paths import project_root
from llm4rec.core.schemas import RecommendationExample, TaskSpec
from llm4rec.workflows.base import BaseWorkflow
from llm4rec.workflows.minionerec.semantic_ids.build import sid_dir
from llm4rec.workflows.minionerec.trainer import SidGRPOTrainer, SidSFTTrainer
from llm4rec.workflows.registry import register_workflow


def _processed_dir(cfg: dict[str, Any]) -> Path:
    root = Path((cfg.get("paths") or {}).get("data_root", "data"))
    if not root.is_absolute():
        root = project_root() / root
    ds = str(cfg["dataset"]["name"])
    return root / "processed" / ds


@register_workflow("minionerec")
class MiniOneRecWorkflow(BaseWorkflow):
    """SID pipeline: prepare → SID build → SFT → GRPO → SID eval."""

    name = "minionerec"

    def __init__(self, context: ExperimentContext | None = None, **kwargs: Any) -> None:
        super().__init__(context, **kwargs)
        self._train_examples: list[RecommendationExample] | None = None
        self._eval_examples: list[RecommendationExample] | None = None
        self._processed: Path | None = None
        self._train_summary: dict[str, Any] = {}

    def required_stages(self) -> list[str]:
        return [
            "prepare_data",
            "build_semantic_ids",
            "build_sid_dataset",
            "sft",
            "grpo",
            "evaluate",
            "report",
        ]

    def prepare_data(self) -> DatasetBundle:
        if self.context is None:
            raise ConfigurationError("context required")
        cfg = self.context.config
        bundle = build_movielens_bundle(cfg["dataset"]["name"], cfg["dataset"], prepare=True)
        processed = _processed_dir(cfg)
        self._processed = processed
        # SID artifacts (built by prepare CLI); expose path in extras
        sid_path = sid_dir(processed)
        extras = {"processed_dir": processed, "sid_dir": sid_path}
        if sid_path.exists():
            extras["semantic_ids"] = sid_path
        self._bundle = bundle.with_extra("semantic_ids", extras.get("semantic_ids"))
        for k, v in extras.items():
            self._bundle = self._bundle.with_extra(k, v)
        return self._bundle

    def build_model(self) -> Any:
        # SID model is prepared inside SidSFTTrainer / SidGRPOTrainer via prepare_sid_model
        return None

    def train(self) -> dict[str, Any]:
        if self.context is None:
            raise ConfigurationError("context required")
        processed = self._processed or _processed_dir(self.context.config)
        stages = list((self.context.config.get("training") or {}).get("stages") or ["sft", "grpo"])
        summary: dict[str, Any] = {}
        sft_path = None
        if "sft" in stages:
            sft = SidSFTTrainer(processed_dir=processed)
            summary["sft"] = sft.train(self.context)
            sft_path = (summary["sft"] or {}).get("adapter_path") or (
                summary["sft"] or {}
            ).get("final_adapter")
        if "grpo" in stages:
            grpo = SidGRPOTrainer(processed_dir=processed, sft_adapter_path=sft_path)
            summary["grpo"] = grpo.train(self.context)
        self._train_summary = summary
        return summary

    def evaluate(self) -> dict[str, Any]:
        if self.context is None:
            raise ConfigurationError("context required")
        processed = self._processed or _processed_dir(self.context.config)
        grpo_sum = (self._train_summary or {}).get("grpo") or {}
        sft_sum = (self._train_summary or {}).get("sft") or {}
        adapter = grpo_sum.get("adapter_path") or grpo_sum.get("final_adapter")
        sft_adapter = sft_sum.get("adapter_path") or sft_sum.get("final_adapter")
        return self.evaluate_sid(
            self.context,
            adapter_path=adapter,
            sft_adapter_path=sft_adapter,
        )

    def inference(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use evaluate_sid / constrained decoding in evaluator")

    def evaluate_sid(
        self,
        context: ExperimentContext,
        *,
        adapter_path: str | None,
        sft_adapter_path: str | None = None,
        split: str = "test",
        tag: str = "test",
    ) -> dict[str, Any]:
        processed = self._processed or _processed_dir(context.config)
        top_k = int(((context.config.get("evaluation") or {}).get("top_k") or [10])[-1])
        max_examples = (context.config.get("evaluation") or {}).get("max_examples")
        free_n = int((context.config.get("evaluation") or {}).get("free_gen_n", 50))
        result = evaluate_sid_checkpoint(
            model_cfg=context.config["model"],
            processed_dir=processed,
            adapter_path=adapter_path,
            sft_adapter_path=sft_adapter_path,
            split=split,
            top_k=top_k,
            max_examples=max_examples,
            free_gen_n=free_n,
            predictions_dir=context.run_dir / "predictions",
        )
        gen = GenerationMetrics.sid_validity(
            [True] * int(result.metrics.get("n", 0) or 0)
        ) if hasattr(result, "metrics") else {}
        payload = {
            "tag": tag,
            "adapter_path": adapter_path,
            "metrics": result.metrics,
            "slices": result.slices,
            "metadata": result.metadata,
            "sid_dir": str(sid_dir(processed)),
            "generation": gen,
        }
        return payload

    # ---- CLI compatibility ----
    def build_examples(self, dataset: Any, split: str) -> list[RecommendationExample]:
        return []

    def build_examples_with_spec(
        self, dataset: Any, split: str, task_spec: TaskSpec
    ) -> list[RecommendationExample]:
        _ = dataset, task_spec
        return []

    def set_examples(
        self,
        train: list[RecommendationExample],
        eval_examples: list[RecommendationExample],
    ) -> None:
        self._train_examples = train
        self._eval_examples = eval_examples

    def build_trainer(self, context: ExperimentContext) -> SidSFTTrainer:
        self.bind(context)
        processed = _processed_dir(context.config)
        self._processed = processed
        return SidSFTTrainer(processed_dir=processed)

    def build_grpo_trainer(
        self, context: ExperimentContext, *, sft_adapter_path: str | None = None
    ) -> SidGRPOTrainer:
        self.bind(context)
        processed = self._processed or _processed_dir(context.config)
        return SidGRPOTrainer(processed_dir=processed, sft_adapter_path=sft_adapter_path)

    def build_evaluator(self, context: ExperimentContext) -> Any:
        self.bind(context)
        return self

    def build_probes(self, context: ExperimentContext) -> list[Any]:
        return []
