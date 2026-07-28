"""DatasetBuilder interface and DatasetBundle contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from llm4rec.core.schemas import Interaction, RecommendationExample


@dataclass
class DatasetBundle:
    """Normalized dataset payload shared across workflows.

    Core fields are always present. Workflow-specific extensions live in
    ``extras`` (e.g. ``semantic_ids``, ``images``, ``captions``, ``candidates``).
    """

    name: str
    interactions: list[Interaction]
    users: list[str]
    items: list[str]
    sequences: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def semantic_ids(self) -> Any:
        return self.extras.get("semantic_ids")

    @property
    def images(self) -> Any:
        return self.extras.get("images")

    @property
    def captions(self) -> Any:
        return self.extras.get("captions")

    @property
    def candidates(self) -> Any:
        return self.extras.get("candidates")

    def with_extra(self, key: str, value: Any) -> "DatasetBundle":
        extras = dict(self.extras)
        extras[key] = value
        return DatasetBundle(
            name=self.name,
            interactions=self.interactions,
            users=self.users,
            items=self.items,
            sequences=dict(self.sequences),
            metadata=dict(self.metadata),
            extras=extras,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_interactions": len(self.interactions),
            "n_users": len(self.users),
            "n_items": len(self.items),
            "n_sequences": len(self.sequences),
            "extra_keys": sorted(self.extras),
            "metadata_keys": sorted(self.metadata),
        }


class DatasetBuilder(ABC):
    """Build a :class:`DatasetBundle` from raw sources / configs."""

    name: str

    @abstractmethod
    def prepare(self) -> None:
        """Download / preprocess raw artifacts if needed."""

    @abstractmethod
    def build(self) -> DatasetBundle:
        """Materialize the bundle."""

    def build_examples(self, split: str, task_spec: Any) -> list[RecommendationExample]:
        """Optional: workflows that use RecommendationExample may override."""
        raise NotImplementedError(f"{type(self).__name__} does not build examples")
