"""Unified evaluation metrics (ranking / bias / generation)."""

from __future__ import annotations

from typing import Any

__all__ = ["RankingMetrics", "BiasMetrics", "GenerationMetrics"]


def __getattr__(name: str) -> Any:
    if name == "RankingMetrics":
        from llm4rec.components.evaluation.ranking import RankingMetrics

        return RankingMetrics
    if name == "BiasMetrics":
        from llm4rec.components.evaluation.bias import BiasMetrics

        return BiasMetrics
    if name == "GenerationMetrics":
        from llm4rec.components.evaluation.generation import GenerationMetrics

        return GenerationMetrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
