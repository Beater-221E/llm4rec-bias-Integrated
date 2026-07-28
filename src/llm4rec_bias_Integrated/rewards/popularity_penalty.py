"""Popularity penalty reward component."""

from __future__ import annotations

import numpy as np
import torch

from llm4rec_bias_Integrated.core.schemas import RewardOutput
from llm4rec_bias_Integrated.prompts.candidate_choice import parse_choice
from llm4rec_bias_Integrated.rewards.base import RewardBatch, RewardContext
from llm4rec_bias_Integrated.rewards.registry import register_reward


@register_reward("popularity_penalty")
class PopularityPenaltyReward:
    """Penalize choosing above-mean popularity candidates (letter-route).

    Returns **negative** lift so the composer adds ``w * component``.
    """

    name = "popularity_penalty"

    def __call__(
        self,
        batch: RewardBatch,
        outputs: list[str],
        context: RewardContext,
    ) -> RewardOutput:
        if batch.pop_quantiles is None:
            raise ValueError("popularity_penalty requires pop_quantiles")
        vals = []
        lifts = []
        for text, quants in zip(outputs, batch.pop_quantiles, strict=True):
            n = len(quants)
            choice = parse_choice(text, n)
            if choice is None:
                vals.append(0.0)
                continue
            lift = float(quants[choice]) - float(np.mean(quants))
            lifts.append(lift)
            vals.append(-lift)  # penalty
        tensor = torch.tensor(vals, dtype=torch.float32)
        return RewardOutput(
            total=tensor,
            components={"popularity_penalty": tensor},
            telemetry={
                "pop_lift": float(np.mean(lifts)) if lifts else 0.0,
                "pop_penalty_mean": float(tensor.mean()),
            },
        )
