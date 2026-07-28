"""Ranking-oriented rewards: accuracy / HR / NDCG."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from llm4rec.components.reward.base import BaseReward, RewardBatch, RewardContext
from llm4rec.components.reward._impl.exact_match import ExactMatchReward
from llm4rec.components.reward._impl.rank_aware import RankAwareReward
from llm4rec.components.reward._impl.registry import register_reward
from llm4rec.components.prompts.candidate_choice import parse_choice
from llm4rec.core.schemas import RewardOutput


@register_reward("accuracy")
class AccuracyReward(BaseReward):
    """Alias of exact letter match (HR@1 on generated choice)."""

    name = "accuracy"

    def __init__(self) -> None:
        self._impl = ExactMatchReward()

    def __call__(
        self,
        batch: RewardBatch,
        outputs: list[str],
        context: RewardContext,
    ) -> RewardOutput:
        return self._impl(batch, outputs, context)


@register_reward("hr")
class HRReward(BaseReward):
    """Hit-rate style reward: 1 if parsed choice equals target else 0."""

    name = "hr"

    def __call__(
        self,
        batch: RewardBatch,
        outputs: list[str],
        context: RewardContext,
    ) -> RewardOutput:
        targets = batch.targets or []
        scores = []
        hits = 0
        for i, text in enumerate(outputs):
            n = len(batch.pop_quantiles[i]) if batch.pop_quantiles else 10
            choice = parse_choice(text, n)
            tgt = int(targets[i]) if i < len(targets) and targets[i] is not None else None
            hit = float(choice is not None and tgt is not None and choice == tgt)
            hits += int(hit)
            scores.append(hit)
        total = torch.tensor(scores, dtype=torch.float32)
        return RewardOutput(
            total=total,
            components={"hr": total},
            telemetry={"hr": hits / max(len(outputs), 1)},
        )


@register_reward("ndcg")
class NDCGReward(BaseReward):
    """NDCG@K proxy from generated letter rank within the candidate list.

    For letter-route GRPO, the model emits one letter; we treat a correct hit
    as rank 1 (NDCG=1) and a miss as NDCG=0. Extensible for scored lists via
    ``batch.extras['ranks']``.
    """

    name = "ndcg"

    def __init__(self, k: int = 10) -> None:
        self.k = int(k)

    def __call__(
        self,
        batch: RewardBatch,
        outputs: list[str],
        context: RewardContext,
    ) -> RewardOutput:
        ranks: Sequence[Any] | None = batch.extras.get("ranks")
        scores: list[float] = []
        if ranks is not None:
            import math

            for rank in ranks:
                r = int(rank)
                if r <= 0 or r > self.k:
                    scores.append(0.0)
                else:
                    scores.append(1.0 / math.log2(r + 1))
        else:
            # letter hit → ndcg 1.0 else 0.0
            targets = batch.targets or []
            for i, text in enumerate(outputs):
                n = len(batch.pop_quantiles[i]) if batch.pop_quantiles else 10
                choice = parse_choice(text, n)
                tgt = int(targets[i]) if i < len(targets) and targets[i] is not None else None
                scores.append(
                    1.0 if choice is not None and tgt is not None and choice == tgt else 0.0
                )
        total = torch.tensor(scores, dtype=torch.float32)
        return RewardOutput(
            total=total,
            components={"ndcg": total},
            telemetry={"ndcg_mean": float(total.mean()) if len(scores) else 0.0},
        )


# Keep legacy names registered
_ = RankAwareReward

__all__ = ["AccuracyReward", "HRReward", "NDCGReward"]
