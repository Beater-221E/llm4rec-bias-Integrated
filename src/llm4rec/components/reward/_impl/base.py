"""Reward function protocol and shared helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import torch

from llm4rec.core.schemas import RewardOutput


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last["content"])
    return str(completion)


@dataclass
class RewardBatch:
    """Batch fields available to reward components."""

    prompts: Sequence[Any]
    completions: Sequence[Any]
    targets: Sequence[Any] | None = None
    pop_quantiles: Sequence[Sequence[float]] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardContext:
    """Optional logging / run context for rewards."""

    log_metric: Any | None = None
    stage: str = "grpo"


class RewardFunction(Protocol):
    name: str

    def __call__(
        self,
        batch: RewardBatch,
        outputs: list[str],
        context: RewardContext,
    ) -> RewardOutput:
        ...
