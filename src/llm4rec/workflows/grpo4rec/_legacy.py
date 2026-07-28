"""GRPO4Rec workflow — candidate-choice SFT + GRPO + log-prob eval."""

from __future__ import annotations

from typing import Any

from llm4rec.core.context import ExperimentContext
from llm4rec.core.exceptions import ConfigurationError
from llm4rec.core.schemas import RecommendationExample, TaskSpec
from llm4rec.components.dataset._impl.base import DatasetAdapter
from llm4rec.components.evaluation._impl.ranking import CandidateLogProbEvaluator
from llm4rec.components.model._impl.loader import load_model_bundle
from llm4rec.components.trainer._impl.distributed import resolve_distributed_plan
from llm4rec.components.trainer._impl.grpo import GRPOLoRATrainer
from llm4rec.components.trainer._impl.sft import SFTLoRATrainer
from llm4rec.workflows.base import RecommendationWorkflow
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
class GRPO4RecWorkflow(RecommendationWorkflow):
    name = "grpo4rec"

    def __init__(self, **_: Any) -> None:
        self._dataset: DatasetAdapter | None = None
        self._train_examples: list[RecommendationExample] | None = None
        self._eval_examples: list[RecommendationExample] | None = None
        self._tok = None
        self._model = None

    def required_stages(self) -> list[str]:
        return ["prepare_data", "optional_sft", "grpo", "evaluate", "analyze", "report"]

    def attach_dataset(self, dataset: DatasetAdapter) -> None:
        self._dataset = dataset

    def build_examples(
        self,
        dataset: DatasetAdapter,
        split: str,
    ) -> list[RecommendationExample]:
        raise ConfigurationError(
            "Call build_examples_with_spec(...) from the train runner"
        )

    def build_examples_with_spec(
        self,
        dataset: DatasetAdapter,
        split: str,
        task_spec: TaskSpec,
    ) -> list[RecommendationExample]:
        self._dataset = dataset
        return dataset.build_examples(split, task_spec)

    def build_model(self, context: ExperimentContext):
        plan = resolve_distributed_plan(
            context.config.get("training") or {},
            model_name=context.model_name,
        )
        tok, model, _precision = load_model_bundle(
            context.config["model"],
            context.config.get("peft"),
            for_training=False,
            local_rank=plan.local_rank,
        )
        self._tok, self._model = tok, model
        return tok, model

    def build_trainer(self, context: ExperimentContext) -> SFTLoRATrainer:
        if self._train_examples is None:
            raise ConfigurationError("train examples not prepared")
        return SFTLoRATrainer(self._train_examples, self._eval_examples)

    def build_grpo_trainer(
        self,
        context: ExperimentContext,
        *,
        sft_adapter_path: str | None = None,
    ) -> GRPOLoRATrainer:
        if self._train_examples is None:
            raise ConfigurationError("train examples not prepared")
        return GRPOLoRATrainer(
            self._train_examples,
            sft_adapter_path=sft_adapter_path,
        )

    def set_examples(
        self,
        train: list[RecommendationExample],
        eval_examples: list[RecommendationExample],
    ) -> None:
        self._train_examples = train
        self._eval_examples = eval_examples

    def build_evaluator(self, context: ExperimentContext) -> CandidateLogProbEvaluator:
        if self._tok is None or self._model is None:
            self.build_model(context)
        top_k = list((context.config.get("evaluation") or {}).get("top_k") or [1, 5, 10])
        return CandidateLogProbEvaluator(
            self._tok,
            self._model,
            top_k=top_k,
            predictions_dir=context.run_dir / "predictions",
        )

    def build_probes(self, context: ExperimentContext) -> list[Any]:
        from llm4rec.components.evaluation.probes.registry import build_probes_from_config

        return build_probes_from_config(context.config.get("bias"))
