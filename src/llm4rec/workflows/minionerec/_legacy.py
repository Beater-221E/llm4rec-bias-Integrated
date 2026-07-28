"""MiniOneRec workflow — semantic-ID generative retrieval (Phase 7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.core.context import ExperimentContext
from llm4rec.core.exceptions import ConfigurationError
from llm4rec.core.paths import project_root
from llm4rec.core.schemas import RecommendationExample, TaskSpec
from llm4rec.components.dataset._impl.base import DatasetAdapter
from llm4rec.components.evaluation._impl.sid import evaluate_sid_checkpoint
from llm4rec.workflows.minionerec.semantic_ids.build import sid_dir
from llm4rec.components.trainer._impl.sid_grpo import SidGRPOTrainer
from llm4rec.components.trainer._impl.sid_sft import SidSFTTrainer
from llm4rec.workflows.base import RecommendationWorkflow
from llm4rec.workflows.registry import register_workflow


def _processed_dir(cfg: dict[str, Any]) -> Path:
    root = Path((cfg.get("paths") or {}).get("data_root", "data"))
    if not root.is_absolute():
        root = project_root() / root
    ds = str(cfg["dataset"]["name"])
    return root / "processed" / ds


@register_workflow("minionerec")
class MiniOneRecWorkflow(RecommendationWorkflow):
    name = "minionerec"

    def __init__(self, **_: Any) -> None:
        self._train_examples: list[RecommendationExample] | None = None
        self._eval_examples: list[RecommendationExample] | None = None
        self._processed: Path | None = None

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

    def build_examples(
        self, dataset: DatasetAdapter, split: str
    ) -> list[RecommendationExample]:
        # SID route uses jsonl under processed/.../sid/; examples optional here
        return []

    def build_examples_with_spec(
        self,
        dataset: DatasetAdapter,
        split: str,
        task_spec: TaskSpec,
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

    def build_model(self, context: ExperimentContext) -> Any:
        return None

    def build_trainer(self, context: ExperimentContext) -> SidSFTTrainer:
        processed = _processed_dir(context.config)
        self._processed = processed
        return SidSFTTrainer(processed_dir=processed)

    def build_grpo_trainer(
        self,
        context: ExperimentContext,
        *,
        sft_adapter_path: str | None = None,
    ) -> SidGRPOTrainer:
        processed = self._processed or _processed_dir(context.config)
        return SidGRPOTrainer(
            processed_dir=processed, sft_adapter_path=sft_adapter_path
        )

    def build_evaluator(self, context: ExperimentContext) -> Any:
        # SID eval is function-based; runner calls evaluate_sid below
        return self

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
        payload = {
            "tag": tag,
            "adapter_path": adapter_path,
            "metrics": result.metrics,
            "slices": result.slices,
            "metadata": result.metadata,
            "sid_dir": str(sid_dir(processed)),
        }
        return payload

    def build_probes(self, context: ExperimentContext) -> list[Any]:
        # letter probes N/A; semantic_prefix lands with SID completions later
        return []
