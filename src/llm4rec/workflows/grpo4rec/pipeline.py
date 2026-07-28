"""GRPO4Rec workflow — Letter candidate-choice SFT + GRPO (behavior-preserving)."""

from __future__ import annotations

from typing import Any

from llm4rec.components.dataset.base import DatasetBundle
from llm4rec.components.dataset.movielens import build_movielens_bundle
from llm4rec.components.evaluation.ranking import CandidateLogProbEvaluator
from llm4rec.components.model.factory import ModelFactory
from llm4rec.components.trainer.distributed_plan import resolve_plan
from llm4rec.components.trainer.grpo import GRPOTrainer
from llm4rec.components.trainer.sft import SFTTrainer
from llm4rec.core.context import ExperimentContext
from llm4rec.core.exceptions import ConfigurationError
from llm4rec.core.schemas import RecommendationExample, TaskSpec
from llm4rec.workflows.base import BaseWorkflow
from llm4rec.workflows.registry import register_workflow


def task_spec_from_config(cfg: dict[str, Any]) -> TaskSpec:
    ds = cfg["dataset"]
    wf = cfg.get("workflow") or {}
    return TaskSpec(
        task=str(wf.get("task") or "candidate_choice"),
        history_max_length=int(ds.get("history_max_length", 20)),
        candidate_size=int(ds.get("candidate_size", 10)),
        negative_sampling=str(ds.get("negative_sampling", "uniform")),
        target_position=str(ds.get("target_position", "random")),
        framing=str(ds.get("framing", "neutral")),
    )


@register_workflow("grpo4rec")
class GRPO4RecWorkflow(BaseWorkflow):
    """Letter-route pipeline: prepare → SFT → GRPO → log-prob eval."""

    name = "grpo4rec"

    def __init__(self, context: ExperimentContext | None = None, **kwargs: Any) -> None:
        super().__init__(context, **kwargs)
        self._train_examples: list[RecommendationExample] | None = None
        self._eval_examples: list[RecommendationExample] | None = None
        self._tok = None
        self._model = None
        self._train_summary: dict[str, Any] = {}

    def required_stages(self) -> list[str]:
        return ["prepare_data", "optional_sft", "grpo", "evaluate", "analyze", "report"]

    def prepare_data(self) -> DatasetBundle:
        if self.context is None:
            raise ConfigurationError("context required")
        cfg = self.context.config
        bundle = build_movielens_bundle(cfg["dataset"]["name"], cfg["dataset"], prepare=True)
        # Build letter examples via legacy adapter for identical behavior
        from llm4rec.components.dataset._impl.registry import build_dataset

        adapter = build_dataset(cfg["dataset"]["name"], cfg["dataset"])
        spec = task_spec_from_config(cfg)
        train = adapter.build_examples("train", spec)
        eval_ex = adapter.build_examples("validation", spec)
        limit_tr = cfg["dataset"].get("train_limit")
        limit_ev = cfg["dataset"].get("eval_limit")
        if limit_tr:
            train = train[: int(limit_tr)]
        if limit_ev:
            eval_ex = eval_ex[: int(limit_ev)]
        self._train_examples = train
        self._eval_examples = eval_ex
        self._bundle = bundle.with_extra("candidates", True)
        return self._bundle

    def build_model(self) -> Any:
        if self.context is None:
            raise ConfigurationError("context required")
        plan = resolve_plan(self.context)
        bundle = ModelFactory.from_config(
            self.context.config["model"],
            self.context.config.get("peft"),
            for_training=False,
            local_rank=plan.local_rank,
        )
        self._tok, self._model = bundle.tokenizer, bundle.model
        return bundle

    def train(self) -> dict[str, Any]:
        if self.context is None:
            raise ConfigurationError("context required")
        if self._train_examples is None:
            self.prepare_data()
        assert self._train_examples is not None
        stages = list((self.context.config.get("training") or {}).get("stages") or ["sft", "grpo"])
        summary: dict[str, Any] = {}
        sft_path = None
        if "sft" in stages:
            sft = SFTTrainer(self._train_examples, self._eval_examples)
            summary["sft"] = sft.train(self.context)
            sft_path = (summary["sft"] or {}).get("adapter_path") or (
                summary["sft"] or {}
            ).get("final_adapter")
        if "grpo" in stages:
            grpo = GRPOTrainer(self._train_examples, sft_adapter_path=sft_path)
            summary["grpo"] = grpo.train(self.context)
        self._train_summary = summary
        return summary

    def evaluate(self) -> dict[str, Any]:
        if self.context is None:
            raise ConfigurationError("context required")
        if self._tok is None or self._model is None:
            self.build_model()
        if self._eval_examples is None:
            self.prepare_data()
        top_k = list((self.context.config.get("evaluation") or {}).get("top_k") or [1, 5, 10])
        evaluator = CandidateLogProbEvaluator(
            self._tok,
            self._model,
            top_k=top_k,
            predictions_dir=self.context.run_dir / "predictions",
        )
        result = evaluator.evaluate(dataset=self._eval_examples, split="validation")
        payload = {
            "metrics": getattr(result, "metrics", result),
            "slices": getattr(result, "slices", {}),
        }
        return payload

    def inference(self, messages: list[dict[str, str]], n_candidates: int = 10) -> Any:
        from llm4rec.components.evaluation.ranking import score_letters
        import torch

        if self._tok is None or self._model is None:
            self.build_model()
        device = next(self._model.parameters()).device
        return score_letters(self._tok, self._model, device, messages, n_candidates)

    # ---- CLI compatibility shims (legacy RecommendationWorkflow API) ----
    def attach_dataset(self, dataset: Any) -> None:
        self._adapter = dataset

    def build_examples(self, dataset: Any, split: str) -> list[RecommendationExample]:
        raise ConfigurationError("Call build_examples_with_spec(...) from the train runner")

    def build_examples_with_spec(
        self, dataset: Any, split: str, task_spec: TaskSpec
    ) -> list[RecommendationExample]:
        return dataset.build_examples(split, task_spec)

    def set_examples(
        self,
        train: list[RecommendationExample],
        eval_examples: list[RecommendationExample],
    ) -> None:
        self._train_examples = train
        self._eval_examples = eval_examples

    def build_trainer(self, context: ExperimentContext) -> SFTTrainer:
        self.bind(context)
        if self._train_examples is None:
            raise ConfigurationError("train examples not prepared")
        return SFTTrainer(self._train_examples, self._eval_examples)

    def build_grpo_trainer(
        self, context: ExperimentContext, *, sft_adapter_path: str | None = None
    ) -> GRPOTrainer:
        self.bind(context)
        if self._train_examples is None:
            raise ConfigurationError("train examples not prepared")
        return GRPOTrainer(self._train_examples, sft_adapter_path=sft_adapter_path)

    def build_evaluator(self, context: ExperimentContext) -> CandidateLogProbEvaluator:
        self.bind(context)
        if self._tok is None or self._model is None:
            # legacy path used load_model_bundle via build_model
            plan = resolve_plan(context)
            bundle = ModelFactory.from_config(
                context.config["model"],
                context.config.get("peft"),
                for_training=False,
                local_rank=plan.local_rank,
            )
            self._tok, self._model = bundle.tokenizer, bundle.model
        top_k = list((context.config.get("evaluation") or {}).get("top_k") or [1, 5, 10])
        return CandidateLogProbEvaluator(
            self._tok,
            self._model,
            top_k=top_k,
            predictions_dir=context.run_dir / "predictions",
        )

    def build_probes(self, context: ExperimentContext) -> list[Any]:
        from llm4rec.components.evaluation.bias import BiasMetrics

        return BiasMetrics.build_probes(context.config.get("bias"))
