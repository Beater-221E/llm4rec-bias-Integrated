"""Dataset schema helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm4rec.core.schemas import Interaction, RecommendationExample, TaskSpec


@dataclass
class InteractionSchema:
    """Declares expected columns / fields for a dataset family."""

    user_field: str = "user_id"
    item_field: str = "item_id"
    rating_field: str = "rating"
    timestamp_field: str = "timestamp"
    required_extras: tuple[str, ...] = ()
    optional_extras: tuple[str, ...] = (
        "semantic_ids",
        "images",
        "captions",
        "candidates",
    )

    def validate_interaction(self, row: Interaction) -> None:
        if not row.user_id or not row.item_id:
            raise ValueError("user_id and item_id are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_field": self.user_field,
            "item_field": self.item_field,
            "rating_field": self.rating_field,
            "timestamp_field": self.timestamp_field,
            "required_extras": list(self.required_extras),
            "optional_extras": list(self.optional_extras),
        }


# Re-export core contracts used by dataset consumers
__all__ = [
    "Interaction",
    "InteractionSchema",
    "RecommendationExample",
    "TaskSpec",
]
