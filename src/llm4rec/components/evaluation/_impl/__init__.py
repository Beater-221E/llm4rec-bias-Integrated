"""Evaluation implementation package (lazy exports)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CandidateLogProbEvaluator",
    "evaluate_predictions",
    "ndcg_at",
    "score_letters",
    "to_evaluation_result",
]


def __getattr__(name: str) -> Any:
    if name in {"CandidateLogProbEvaluator", "ndcg_at", "score_letters"}:
        from llm4rec.components.evaluation._impl import ranking as _r

        return getattr(_r, name)
    if name in {"evaluate_predictions", "to_evaluation_result"}:
        from llm4rec.components.evaluation._impl import adapter as _a

        return getattr(_a, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
