"""Plugin reward base types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llm4rec.components.reward._impl.base import (
    RewardBatch,
    RewardContext,
    RewardFunction,
    completion_text,
)
from llm4rec.core.schemas import RewardOutput


class BaseReward(ABC):
    """Pluggable reward component.

    New rewards register via ``@register_reward`` and are composed from config
    without modifying model or trainer code.
    """

    name: str = "base"

    @abstractmethod
    def __call__(
        self,
        batch: RewardBatch,
        outputs: list[str],
        context: RewardContext,
    ) -> RewardOutput:
        ...


__all__ = [
    "BaseReward",
    "RewardBatch",
    "RewardContext",
    "RewardFunction",
    "RewardOutput",
    "completion_text",
]
