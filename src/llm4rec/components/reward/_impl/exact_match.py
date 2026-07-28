"""Exact-match reward for letter candidate choice."""

from __future__ import annotations

import torch

from llm4rec.core.schemas import RewardOutput
from llm4rec.components.prompts.candidate_choice import parse_choice
from llm4rec.components.reward._impl.base import RewardBatch, RewardContext, completion_text
from llm4rec.components.reward._impl.registry import register_reward

REWARD_HIT = 1.0
REWARD_MISS = 0.0


@register_reward("exact_match")
class ExactMatchReward:
    name = "exact_match"

    def __call__(
        self,
        batch: RewardBatch,
        outputs: list[str],
        context: RewardContext,
    ) -> RewardOutput:
        if batch.targets is None:
            raise ValueError("exact_match requires targets")
        vals = []
        hits = 0
        valid = 0
        for text, target, quants in zip(
            outputs,
            batch.targets,
            batch.pop_quantiles or [None] * len(outputs),
            strict=True,
        ):
            n = len(quants) if quants is not None else 10
            choice = parse_choice(text, n)
            if choice is None:
                vals.append(0.0)  # format component owns invalid penalty
                continue
            valid += 1
            hit = choice == int(target)
            hits += int(hit)
            vals.append(REWARD_HIT if hit else REWARD_MISS)
        tensor = torch.tensor(vals, dtype=torch.float32)
        return RewardOutput(
            total=tensor,
            components={"exact_match": tensor},
            telemetry={
                "exact_hit_rate": hits / max(valid, 1),
                "n_valid": float(valid),
            },
        )
