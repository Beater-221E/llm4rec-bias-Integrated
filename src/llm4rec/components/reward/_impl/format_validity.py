"""Format / parse-validity reward (shared parser with evaluation)."""

from __future__ import annotations

import torch

from llm4rec.core.schemas import RewardOutput
from llm4rec.components.prompts.candidate_choice import parse_choice
from llm4rec.components.reward._impl.base import RewardBatch, RewardContext
from llm4rec.components.reward._impl.registry import register_reward

REWARD_VALID = 1.0
REWARD_INVALID = -0.5  # upstream letter-route default; must stay dominated by wrong-valid when combined


@register_reward("format_validity")
class FormatValidityReward:
    name = "format_validity"

    def __init__(self, invalid_penalty: float = REWARD_INVALID) -> None:
        self.invalid_penalty = float(invalid_penalty)

    def __call__(
        self,
        batch: RewardBatch,
        outputs: list[str],
        context: RewardContext,
    ) -> RewardOutput:
        vals = []
        invalid = 0
        for text, quants in zip(
            outputs,
            batch.pop_quantiles or [None] * len(outputs),
            strict=True,
        ):
            n = len(quants) if quants is not None else 10
            choice = parse_choice(text, n)
            if choice is None:
                vals.append(self.invalid_penalty)
                invalid += 1
            else:
                vals.append(REWARD_VALID)
        tensor = torch.tensor(vals, dtype=torch.float32)
        return RewardOutput(
            total=tensor,
            components={"format_validity": tensor},
            telemetry={"invalid_rate": invalid / max(len(outputs), 1)},
        )
