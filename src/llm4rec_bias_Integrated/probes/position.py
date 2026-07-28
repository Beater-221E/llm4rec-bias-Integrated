"""Position probe: place target at every candidate slot (upstream-compatible)."""

from __future__ import annotations

from typing import Any

import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llm4rec_bias_Integrated.core.schemas import ProbeResult, RecommendationExample
from llm4rec_bias_Integrated.probes.base import BiasProbe
from llm4rec_bias_Integrated.probes.rebuild import place_target_at_slot
from llm4rec_bias_Integrated.probes.registry import register_probe
from llm4rec_bias_Integrated.probes.scoring import choose_index


def _kendall_tau(xs: list[float], ys: list[float]) -> float | None:
    """Simple Kendall tau-b on paired ranks (no ties expected for slots)."""
    n = len(xs)
    if n < 2:
        return None
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 or dy == 0:
                continue
            if dx * dy > 0:
                conc += 1
            else:
                disc += 1
    denom = conc + disc
    if denom == 0:
        return None
    return float((conc - disc) / denom)


@register_probe("position")
class PositionProbe(BiasProbe):
    name = "position"

    def __init__(
        self,
        *,
        place_target_at_all_slots: bool = True,
        compute_spread: bool = True,
        compute_kendall: bool = True,
        n: int | None = None,
        **_: Any,
    ) -> None:
        self.place_target_at_all_slots = bool(place_target_at_all_slots)
        self.compute_spread = bool(compute_spread)
        self.compute_kendall = bool(compute_kendall)
        self.n = n

    def run(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        examples: list[RecommendationExample],
        *,
        device: Any,
        cfg: dict[str, Any] | None = None,
    ) -> ProbeResult:
        cfg = cfg or {}
        n_limit = self.n if self.n is not None else cfg.get("n")
        sub = examples[: int(n_limit)] if n_limit else examples
        if not sub:
            return ProbeResult(name=self.name, metrics={"n": 0.0}, details={})

        c = len(sub[0].candidates or sub[0].features.get("candidate_titles") or [])
        acc = np.zeros(c, dtype=np.float64)
        # For Kendall: does higher accuracy associate with earlier slots?
        slot_acc_pairs: list[tuple[float, float]] = []

        for ex in sub:
            for pos in range(c):
                cf = place_target_at_slot(ex, pos)
                choice = choose_index(tokenizer, model, device, cf)
                hit = float(choice == pos)
                acc[pos] += hit

        acc /= max(len(sub), 1)
        for pos in range(c):
            slot_acc_pairs.append((float(pos), float(acc[pos])))

        metrics: dict[str, float] = {
            "n": float(len(sub)),
            "n_slots": float(c),
        }
        if self.compute_spread:
            metrics["spread"] = float(acc.max() - acc.min())
        if self.compute_kendall:
            tau = _kendall_tau(
                [p for p, _ in slot_acc_pairs],
                [a for _, a in slot_acc_pairs],
            )
            if tau is not None:
                metrics["kendall_tau"] = tau

        # Flatten slot accuracies into metrics for easy aggregation
        for i, v in enumerate(acc.tolist()):
            metrics[f"acc_at_slot_{i}"] = float(v)

        return ProbeResult(
            name=self.name,
            metrics=metrics,
            details={
                "acc_by_target_pos": [round(float(x), 4) for x in acc.tolist()],
                "place_target_at_all_slots": self.place_target_at_all_slots,
            },
        )
