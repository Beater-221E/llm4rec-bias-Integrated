"""Unified BaseWorkflow interface for independent paper pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llm4rec.components.dataset.base import DatasetBundle
from llm4rec.core.context import ExperimentContext


class BaseWorkflow(ABC):
    """Independent recommendation workflow contract.

    Each paper method (MiniOneRec / MLLM4Rec / GRPO4Rec) implements this
    interface without sharing algorithm logic across subclasses.
    """

    name: str

    def __init__(self, context: ExperimentContext | None = None, **_: Any) -> None:
        self.context = context
        self._bundle: DatasetBundle | None = None
        self._model: Any = None

    def bind(self, context: ExperimentContext) -> "BaseWorkflow":
        self.context = context
        return self

    @abstractmethod
    def prepare_data(self) -> DatasetBundle:
        ...

    @abstractmethod
    def build_model(self) -> Any:
        ...

    @abstractmethod
    def train(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def evaluate(self) -> dict[str, Any]:
        ...

    def inference(self, *args: Any, **kwargs: Any) -> Any:
        """Optional generation / ranking inference hook."""
        raise NotImplementedError(f"{self.name} does not implement inference()")

    def required_stages(self) -> list[str]:
        return ["prepare_data", "build_model", "train", "evaluate"]

    def run(self) -> dict[str, Any]:
        """Execute the standard lifecycle."""
        if self.context is None:
            raise RuntimeError("Workflow context is not bound")
        bundle = self.prepare_data()
        self._bundle = bundle
        self._model = self.build_model()
        train_summary = self.train()
        eval_summary = self.evaluate()
        return {"train": train_summary, "evaluate": eval_summary, "data": bundle.summary()}


# Legacy RecommendationWorkflow kept for CLI adapters
from llm4rec.workflows._legacy_base import RecommendationWorkflow  # noqa: E402,F401

__all__ = ["BaseWorkflow", "RecommendationWorkflow"]
