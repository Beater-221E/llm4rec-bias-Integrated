"""Framing probe: same content under text framing variants."""

from __future__ import annotations

from typing import Any

import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llm4rec_bias_Integrated.core.schemas import ProbeResult, RecommendationExample
from llm4rec_bias_Integrated.datasets.transforms.framing import FRAMING_VARIANTS, framing_gap_label
from llm4rec_bias_Integrated.probes.base import BiasProbe
from llm4rec_bias_Integrated.probes.rebuild import rebuild_example
from llm4rec_bias_Integrated.probes.registry import register_probe
from llm4rec_bias_Integrated.probes.scoring import score_example


@register_probe("framing")
class FramingProbe(BiasProbe):
    name = "framing"

    def __init__(
        self,
        *,
        variants: list[str] | None = None,
        **_: Any,
    ) -> None:
        default = ["neutral", "evaluative"]
        self.variants = [v for v in (variants or default) if v in FRAMING_VARIANTS]
        if not self.variants:
            self.variants = default

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
        per_variant: dict[str, dict[str, list[float]]] = {
            v: {"hits": [], "lifts": []} for v in self.variants
        }

        for ex in examples:
            for variant in self.variants:
                cf = rebuild_example(
                    ex,
                    framing=variant,
                    extra_features={"probe_framing": variant},
                    example_id_suffix=f":frame_{variant}",
                )
                row = score_example(tokenizer, model, device, cf)
                per_variant[variant]["hits"].append(row["hit"])
                per_variant[variant]["lifts"].append(row["pop_lift"])

        metrics: dict[str, float] = {"n": float(len(examples))}
        hr_by: dict[str, float] = {}
        for variant, series in per_variant.items():
            hr = float(np.mean(series["hits"])) if series["hits"] else 0.0
            lift = float(np.mean(series["lifts"])) if series["lifts"] else 0.0
            metrics[f"hr@1_{variant}"] = hr
            metrics[f"pop_lift_{variant}"] = lift
            hr_by[variant] = hr

        # Pairwise HR gaps
        for i, a in enumerate(self.variants):
            for b in self.variants[i + 1 :]:
                label = framing_gap_label(a, b)
                metrics[f"gap_{label}"] = float(hr_by[a] - hr_by[b])

        return ProbeResult(
            name=self.name,
            metrics=metrics,
            details={"variants": self.variants},
        )
