"""Position bias penalty reward."""

from __future__ import annotations

import torch

from llm4rec.components.reward.base import BaseReward, RewardBatch, RewardContext
from llm4rec.components.reward._impl.registry import register_reward
from llm4rec.components.prompts.candidate_choice import parse_choice
from llm4rec.core.schemas import RewardOutput


@register_reward("position_penalty")
@register_reward("position")
class PositionBiasPenalty(BaseReward):
    """Penalize choosing early list positions (position bias shortcut).

    Invalid parses receive ``invalid_penalty``.
    """

    name = "position_penalty"

    def __init__(self, invalid_penalty: float = -0.5, weight_scale: float = 1.0) -> None:
        self.invalid_penalty = float(invalid_penalty)
        self.weight_scale = float(weight_scale)

    def __call__(
        self,
        batch: RewardBatch,
        outputs: list[str],
        context: RewardContext,
    ) -> RewardOutput:
        scores: list[float] = []
        positions: list[float] = []
        for i, text in enumerate(outputs):
            n = len(batch.pop_quantiles[i]) if batch.pop_quantiles else 10
            choice = parse_choice(text, n)
            if choice is None:
                scores.append(self.invalid_penalty)
                continue
            pos = choice / max(n - 1, 1)
            positions.append(pos)
            scores.append(-self.weight_scale * (1.0 - pos))
        total = torch.tensor(scores, dtype=torch.float32)
        return RewardOutput(
            total=total,
            components={"position_penalty": total},
            telemetry={
                "position_mean": float(sum(positions) / len(positions)) if positions else 0.0
            },
        )


__all__ = ["PositionBiasPenalty"]
