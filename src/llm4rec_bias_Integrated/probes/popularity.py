"""Observational popularity / exposure probe."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llm4rec_bias_Integrated.core.schemas import ProbeResult, RecommendationExample
from llm4rec_bias_Integrated.probes.base import BiasProbe
from llm4rec_bias_Integrated.probes.registry import register_probe
from llm4rec_bias_Integrated.probes.scoring import score_example


@register_probe("popularity")
class PopularityProbe(BiasProbe):
    name = "popularity"

    def __init__(
        self,
        *,
        tiers: list[str] | None = None,
        report_user_anchored_delta_gap: bool = True,
        report_catalog_pop_lift: bool = True,
        **_: Any,
    ) -> None:
        self.tiers = list(tiers or ["head", "mid", "tail"])
        self.report_user_anchored_delta_gap = bool(report_user_anchored_delta_gap)
        self.report_catalog_pop_lift = bool(report_catalog_pop_lift)

    def run(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        examples: list[RecommendationExample],
        *,
        device: Any,
        cfg: dict[str, Any] | None = None,
    ) -> ProbeResult:
        _ = cfg
        hits, lifts, deltas = [], [], []
        pos_hist: Counter[int] = Counter()
        tier_counts: Counter[str] = Counter()

        for ex in examples:
            row = score_example(tokenizer, model, device, ex)
            hits.append(row["hit"])
            lifts.append(row["pop_lift"])
            pos_hist[int(row["choice"])] += 1
            tier_counts[str(row["tier"])] += 1
            if self.report_user_anchored_delta_gap:
                quants = ex.features.get("pop_quantiles") or []
                hist_mean = float(ex.features.get("history_popularity_mean") or 0.5)
                if quants:
                    deltas.append(float(quants[row["choice"]]) - hist_mean)

        n = max(len(examples), 1)
        metrics: dict[str, float] = {
            "n": float(len(examples)),
            "hr@1": float(np.mean(hits)) if hits else 0.0,
            "pop_lift": float(np.mean(lifts)) if lifts else 0.0,
        }
        if self.report_catalog_pop_lift:
            metrics["catalog_pop_lift"] = metrics["pop_lift"]
        if self.report_user_anchored_delta_gap and deltas:
            metrics["delta_gap"] = float(np.mean(deltas))

        for tier in self.tiers:
            metrics[f"tier_choice_rate_{tier}"] = float(tier_counts.get(tier, 0) / n)

        return ProbeResult(
            name=self.name,
            metrics=metrics,
            details={"chosen_pos_hist": {str(k): int(v) for k, v in sorted(pos_hist.items())}},
        )
