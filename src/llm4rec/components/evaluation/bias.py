"""Bias evaluation metrics and probe registry facade."""

from __future__ import annotations

from typing import Any

from llm4rec.components.evaluation.probes.registry import (
    build_probes_from_config,
    register_probe,
)
from llm4rec.compatibility.llm4rec_bias_eval import letter_pop_lift


class BiasMetrics:
    """Compute popularity / position / framing / history bias summaries.

    Probe implementations live under ``components/evaluation/probes/`` and are
    selected via config — workflows do not hard-code probe classes.
    """

    METRIC_KEYS = (
        "popularity_bias",
        "position_bias",
        "framing_bias",
        "history_bias",
    )

    @staticmethod
    def popularity_lift(pop_quantiles: list[float] | tuple[float, ...], chosen_index: int) -> float:
        return float(letter_pop_lift(pop_quantiles, chosen_index))

    @staticmethod
    def build_probes(bias_cfg: dict[str, Any] | None) -> list[Any]:
        return build_probes_from_config(bias_cfg)

    @staticmethod
    def empty_report() -> dict[str, Any]:
        return {k: None for k in BiasMetrics.METRIC_KEYS}


__all__ = ["BiasMetrics", "build_probes_from_config", "register_probe", "letter_pop_lift"]
