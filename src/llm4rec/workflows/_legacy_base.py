"""Workflow interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llm4rec.core.context import ExperimentContext
from llm4rec.core.schemas import RecommendationExample
from llm4rec.components.dataset._impl.base import DatasetAdapter


class RecommendationWorkflow(ABC):
    """Compose dataset → examples → trainer → evaluator → probes."""

    name: str

    @abstractmethod
    def required_stages(self) -> list[str]:
        ...

    @abstractmethod
    def build_examples(
        self,
        dataset: DatasetAdapter,
        split: str,
    ) -> list[RecommendationExample]:
        ...

    @abstractmethod
    def build_model(self, context: ExperimentContext) -> Any:
        ...

    @abstractmethod
    def build_trainer(self, context: ExperimentContext) -> Any:
        ...

    @abstractmethod
    def build_evaluator(self, context: ExperimentContext) -> Any:
        ...

    @abstractmethod
    def build_probes(self, context: ExperimentContext) -> list[Any]:
        ...
