"""History window transforms."""

from __future__ import annotations

from llm4rec_bias_Integrated.core.schemas import Interaction


def truncate_history(
    history: list[Interaction],
    max_length: int,
) -> list[Interaction]:
    """Keep the most recent ``max_length`` interactions."""
    if max_length <= 0:
        return []
    return history[-max_length:]


def history_item_ids(history: list[Interaction]) -> list[str]:
    return [ix.item_id for ix in history]
