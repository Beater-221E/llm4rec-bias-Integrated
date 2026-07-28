"""Recency / history-order probe."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llm4rec_bias_Integrated.core.exceptions import ConfigurationError
from llm4rec_bias_Integrated.core.schemas import ProbeResult, RecommendationExample
from llm4rec_bias_Integrated.probes.base import BiasProbe
from llm4rec_bias_Integrated.probes.rebuild import rebuild_example
from llm4rec_bias_Integrated.probes.registry import register_probe
from llm4rec_bias_Integrated.probes.scoring import score_example


def _apply_intervention(
    titles: list[str],
    item_ids: list[str],
    name: str,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    if len(titles) != len(item_ids):
        raise ConfigurationError("history titles/ids length mismatch")
    if name == "reverse_history":
        return list(reversed(titles)), list(reversed(item_ids))
    if name == "shuffle_history":
        order = list(range(len(titles)))
        rng.shuffle(order)
        return [titles[i] for i in order], [item_ids[i] for i in order]
    if name == "recent_window":
        k = max(1, len(titles) // 2)
        return titles[-k:], item_ids[-k:]
    if name == "early_window":
        k = max(1, len(titles) // 2)
        return titles[:k], item_ids[:k]
    if name == "duplicate_last":
        if not titles:
            return titles, item_ids
        return titles + [titles[-1]], item_ids + [item_ids[-1]]
    if name == "remove_last":
        if len(titles) <= 1:
            return titles, item_ids
        return titles[:-1], item_ids[:-1]
    raise ConfigurationError(f"Unknown recency intervention '{name}'")


@register_probe("recency")
class RecencyProbe(BiasProbe):
    name = "recency"

    def __init__(
        self,
        *,
        interventions: list[str] | None = None,
        seed: int = 0,
        **_: Any,
    ) -> None:
        self.interventions = list(
            interventions
            or ["reverse_history", "shuffle_history"]
        )
        self.seed = int(seed)

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
        rng = random.Random(self.seed)
        baseline_hits: list[float] = []
        per_iv: dict[str, list[float]] = {iv: [] for iv in self.interventions}

        for ex in examples:
            base = score_example(tokenizer, model, device, ex)
            baseline_hits.append(base["hit"])
            titles = list(ex.features["history_titles"])
            ids = list(ex.history_item_ids)
            for iv in self.interventions:
                new_titles, new_ids = _apply_intervention(titles, ids, iv, rng)
                if len(new_titles) < 1:
                    continue
                cf = rebuild_example(
                    ex,
                    history_titles=new_titles,
                    history_item_ids=new_ids,
                    extra_features={"probe_recency": iv},
                    example_id_suffix=f":rec_{iv}",
                )
                row = score_example(tokenizer, model, device, cf)
                per_iv[iv].append(row["hit"])

        base_hr = float(np.mean(baseline_hits)) if baseline_hits else 0.0
        metrics: dict[str, float] = {
            "n": float(len(examples)),
            "hr@1_original": base_hr,
        }
        details: dict[str, Any] = {"interventions": self.interventions}
        for iv, hits in per_iv.items():
            hr = float(np.mean(hits)) if hits else 0.0
            metrics[f"hr@1_{iv}"] = hr
            metrics[f"delta_hr@1_{iv}"] = hr - base_hr

        return ProbeResult(name=self.name, metrics=metrics, details=details)
