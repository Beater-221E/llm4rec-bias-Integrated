"""Ranking metrics: HR / Recall / NDCG / MRR."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def hit_at_k(rank: int | None, k: int) -> float:
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 if rank <= k else 0.0


def recall_at_k(rank: int | None, k: int) -> float:
    # single-target recall ≡ hit
    return hit_at_k(rank, k)


def ndcg_at_k(rank: int | None, k: int) -> float:
    if rank is None or rank <= 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def mrr(rank: int | None) -> float:
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / float(rank)


class RankingMetrics:
    """Aggregate ranking metrics over a list of target ranks (1-indexed)."""

    def __init__(self, top_k: Sequence[int] | None = None) -> None:
        self.top_k = sorted(top_k or [1, 5, 10])

    def evaluate(self, ranks: Iterable[int | None]) -> dict[str, float]:
        ranks_list = list(ranks)
        n = max(len(ranks_list), 1)
        out: dict[str, float] = {}
        for k in self.top_k:
            out[f"HR@{k}"] = sum(hit_at_k(r, k) for r in ranks_list) / n
            out[f"Recall@{k}"] = sum(recall_at_k(r, k) for r in ranks_list) / n
            out[f"NDCG@{k}"] = sum(ndcg_at_k(r, k) for r in ranks_list) / n
        out["MRR"] = sum(mrr(r) for r in ranks_list) / n
        return out


# Re-export letter-route evaluator for workflows
from llm4rec.components.evaluation._impl.ranking import (  # noqa: E402,F401
    CandidateLogProbEvaluator,
    ndcg_at,
    score_letters,
)

__all__ = [
    "RankingMetrics",
    "CandidateLogProbEvaluator",
    "hit_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "ndcg_at",
    "mrr",
    "score_letters",
]
