"""Dataset adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llm4rec_bias_Integrated.core.schemas import (
    DatasetSplits,
    Interaction,
    RecommendationExample,
    TaskSpec,
)


class DatasetAdapter(ABC):
    """Read, normalize, split, and materialize recommendation examples."""

    name: str

    @abstractmethod
    def download(self) -> None:
        """Fetch raw files into the configured raw directory."""

    @abstractmethod
    def preprocess(self) -> None:
        """Normalize interactions, metadata, and caches."""

    @abstractmethod
    def load_interactions(self) -> list[Interaction]:
        """Return normalized positive (or filtered) interactions."""

    @abstractmethod
    def build_splits(self) -> DatasetSplits:
        """Build train/validation/test partitions."""

    @abstractmethod
    def build_examples(
        self,
        split: str,
        task_spec: TaskSpec,
    ) -> list[RecommendationExample]:
        """Materialize standardized examples for a split + task."""

    @abstractmethod
    def fingerprint(self) -> str:
        """Stable hash of dataset contents and preprocessing knobs."""

    def summary(self) -> dict[str, Any]:
        """Optional human-readable stats; adapters may override."""
        return {"name": self.name}
