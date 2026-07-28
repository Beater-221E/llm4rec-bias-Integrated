"""Rank-aware reward — group-frequency approximation (compatibility).

MiniOneRec uses constrained-beam rank; for free-sample GRPO we approximate
rank by how often an answer appears within the generation group
(``compatibility approximation``).
"""

from __future__ import annotations

from collections import Counter

import torch

from llm4rec.core.schemas import RewardOutput
from llm4rec.components.prompts.candidate_choice import parse_choice
from llm4rec.components.reward._impl.base import RewardBatch, RewardContext
from llm4rec.components.reward._impl.registry import register_reward


@register_reward("rank_aware")
class RankAwareReward:
    name = "rank_aware"

    def __init__(self, group_size: int | None = None) -> None:
        self.group_size = group_size

    def __call__(
        self,
        batch: RewardBatch,
        outputs: list[str],
        context: RewardContext,
    ) -> RewardOutput:
        # Parse choices; invalid → 0 contribution (format owns the penalty)
        choices: list[int | None] = []
        for text, quants in zip(
            outputs,
            batch.pop_quantiles or [None] * len(outputs),
            strict=True,
        ):
            n = len(quants) if quants is not None else 10
            choices.append(parse_choice(text, n))

        # Frequency within the whole batch as a proxy for group rank pressure.
        # When group_size is set, score within contiguous groups.
        vals = [0.0] * len(choices)
        g = self.group_size or len(choices)
        for start in range(0, len(choices), g):
            chunk = choices[start : start + g]
            freq = Counter(c for c in chunk if c is not None)
            for i, c in enumerate(chunk):
                if c is None:
                    vals[start + i] = 0.0
                else:
                    # More frequent wrong answers get stronger penalty
                    # Correctness is handled by exact_match; here we only shape exploration.
                    vals[start + i] = -0.1 * float(freq[c] - 1)
        tensor = torch.tensor(vals, dtype=torch.float32)
        return RewardOutput(
            total=tensor,
            components={"rank_aware": tensor},
            telemetry={"rank_penalty_mean": float(tensor.mean())},
        )
