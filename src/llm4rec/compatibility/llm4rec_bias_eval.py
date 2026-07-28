"""Compatibility bridge to dragonfly90/llm4rec-bias evaluation definitions.

This module ports the *pure metric functions* from the upstream repo so that
lab metrics stay bit-compatible. Model I/O stays in this lab; only the metric
math is shared.

Upstream references (local checkout expected at ``../llm4rec-bias`` when present):
  - ``src/llm4rec/eval.py``      — letter-route ``ndcg_at``, ``evaluate`` aggregation
  - ``src/llm4rec/sid_eval.py``  — ``gini``, IPS / exposure / tier metrics

Metric provenance is tagged per key:
  - ``source=upstream``  — identical definition to llm4rec-bias
  - ``source=extended``  — lab extension (e.g. HR@5/10, MRR on letter route)
  - value ``\"not_applicable\"`` — must never be silently zero-filled
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

NOT_APPLICABLE = "not_applicable"

# Bump when ported formulas change (regression tests pin this).
UPSTREAM_EVAL_VERSION = {
    "compat_module": "llm4rec.compatibility.llm4rec_bias_eval",
    "compat_version": "1.0.0",
    "upstream_repo": "https://github.com/dragonfly90/llm4rec-bias",
    "ported_from": {
        "ndcg_at": "src/llm4rec/eval.py",
        "gini": "src/llm4rec/sid_eval.py",
        "ips": "src/llm4rec/sid_eval.py (inline)",
        "pop_lift_letter": "src/llm4rec/eval.py",
        "pop_lift_catalog": "src/llm4rec/sid_eval.py",
        "delta_gap": "src/llm4rec/sid_eval.py",
        "coverage": "src/llm4rec/sid_eval.py",
        "hr_by_tier": "src/llm4rec/sid_eval.py",
    },
}


# ---------------------------------------------------------------------------
# Pure functions (bit-compatible with upstream)
# ---------------------------------------------------------------------------


def ndcg_at(rank: int, k: int = 5) -> float:
    """Upstream ``eval.ndcg_at`` — DCG with gain 1 at ``rank`` if ``rank < k``."""
    return float(1.0 / np.log2(rank + 2)) if rank < k else 0.0


def gini(counts: np.ndarray | Sequence[float]) -> float:
    """Upstream ``sid_eval.gini`` — zeros included (full-catalog exposure)."""
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(x)
    total = x.sum()
    if n == 0 or total == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * total) - (n + 1) / n)


def letter_pop_lift(pop_quantiles: Sequence[float], chosen_index: int) -> float:
    """Upstream letter-route pop_lift: q[choice] − mean(candidate quantiles)."""
    q = np.asarray(pop_quantiles, dtype=np.float64)
    return float(q[chosen_index] - float(np.mean(q)))


def catalog_pop_lift(top1_quantile: float, catalog_mean_quantile: float) -> float:
    """Upstream SID ``pop_lift@1``: q(top-1) − catalog mean (≈0.5)."""
    return float(top1_quantile - catalog_mean_quantile)


def delta_gap(top1_quantile: float, hist_pop_mean: float) -> float:
    """Upstream ΔGAP: q(top-1) − user history popularity mean."""
    return float(top1_quantile - hist_pop_mean)


def ips_weight(count: int | float, gamma: float = 1.0) -> float:
    """Upstream propensity weight: ``1 / max(count, 1)^gamma``."""
    return float(1.0 / (max(float(count), 1.0) ** gamma))


def snips(
    weights: Sequence[float],
    weighted_values: Sequence[float],
) -> float | None:
    """Self-normalized IPS: Σ w·v / Σ w. ``None`` if denominator is 0."""
    denom = float(sum(weights))
    if denom == 0.0:
        return None
    return float(sum(weighted_values) / denom)


def popularity_tier(quantile: float) -> str:
    """Upstream head/mid/tail split on target quantile thirds."""
    if quantile < 1.0 / 3.0:
        return "tail"
    if quantile < 2.0 / 3.0:
        return "mid"
    return "head"


def coverage_at(exposure_counts: Sequence[float]) -> float:
    """Upstream ``coverage@K``: fraction of catalog with exposure > 0."""
    vec = np.asarray(exposure_counts, dtype=np.float64)
    if len(vec) == 0:
        return 0.0
    return float((vec > 0).sum() / len(vec))


def hit_at(rank: int | None, k: int) -> float:
    if rank is None:
        return 0.0
    return 1.0 if rank < k else 0.0


def mrr_at(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return float(1.0 / (rank + 1))


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


@dataclass
class RankedExample:
    """Minimal per-example ranking record for metric aggregation."""

    target_rank: int | None  # 0-based; None = miss beyond list
    chosen_index: int | None = None
    pop_quantiles: list[float] | None = None  # letter-route candidate quantiles
    top1_pop_quantile: float | None = None  # SID / full-catalog
    hist_pop_mean: float | None = None
    target_pop_quantile: float | None = None
    target_count: int | None = None
    exposed_item_ids: list[str] | None = None  # top-K items for exposure
    free_gen_valid: bool | None = None
    constrained_valid: bool | None = None


@dataclass
class CompatMetrics:
    """Aggregated metrics with provenance tags."""

    metrics: dict[str, float | str]
    slices: dict[str, dict[str, float | str | int]] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    version: dict[str, Any] = field(default_factory=lambda: dict(UPSTREAM_EVAL_VERSION))


def _set(
    out: CompatMetrics,
    key: str,
    value: float | str | None,
    *,
    source: str,
) -> None:
    if value is None:
        out.metrics[key] = NOT_APPLICABLE
    else:
        out.metrics[key] = value
    out.sources[key] = source


def aggregate_letter_route(
    examples: Sequence[RankedExample],
    *,
    top_k: Sequence[int] = (1, 5, 10),
) -> CompatMetrics:
    """Aggregate letter-choice ranking metrics (candidate shortlist)."""
    out = CompatMetrics(metrics={"n": float(len(examples))})
    out.sources["n"] = "upstream"

    if not examples:
        for k in top_k:
            _set(out, f"hr@{k}", NOT_APPLICABLE, source="upstream" if k == 1 else "extended")
            _set(out, f"ndcg@{k}", NOT_APPLICABLE, source="upstream" if k == 5 else "extended")
        _set(out, "mrr", NOT_APPLICABLE, source="extended")
        _set(out, "pop_lift", NOT_APPLICABLE, source="upstream")
        for key in (
            "hr_ips@10",
            "ndcg_ips@10",
            "pop_lift@1",
            "delta_gap",
            "exposure_gini",
            "coverage@10",
            "free_gen_valid_rate",
            "constrained_gen_valid_rate",
        ):
            _set(out, key, NOT_APPLICABLE, source="upstream")
        return out

    for k in top_k:
        hits = [hit_at(ex.target_rank, k) for ex in examples]
        ndcgs = [
            ndcg_at(ex.target_rank, k) if ex.target_rank is not None else 0.0
            for ex in examples
        ]
        src_hr = "upstream" if k == 1 else "extended"
        src_ndcg = "upstream" if k == 5 else "extended"
        _set(out, f"hr@{k}", float(np.mean(hits)), source=src_hr)
        _set(out, f"ndcg@{k}", float(np.mean(ndcgs)), source=src_ndcg)

    _set(
        out,
        "mrr",
        float(np.mean([mrr_at(ex.target_rank) for ex in examples])),
        source="extended",
    )

    lifts = []
    chosen = []
    for ex in examples:
        if ex.pop_quantiles is not None and ex.chosen_index is not None:
            lifts.append(letter_pop_lift(ex.pop_quantiles, ex.chosen_index))
            chosen.append(ex.chosen_index)
    _set(out, "pop_lift", float(np.mean(lifts)) if lifts else None, source="upstream")
    if chosen:
        out.slices["chosen_pos_hist"] = {
            str(k): int(v) for k, v in Counter(chosen).items()
        }

    # Full-catalog / SID-only metrics are N/A on letter route
    for key, src in (
        ("hr_ips@10", "upstream"),
        ("ndcg_ips@10", "upstream"),
        ("pop_lift@1", "upstream"),
        ("delta_gap", "upstream"),
        ("exposure_gini", "upstream"),
        ("coverage@10", "upstream"),
        ("free_gen_valid_rate", "upstream"),
        ("constrained_gen_valid_rate", "extended"),
        ("eval_loss", "extended"),
        ("token_accuracy", "extended"),
    ):
        _set(out, key, NOT_APPLICABLE, source=src)
    return out


def aggregate_full_catalog(
    examples: Sequence[RankedExample],
    *,
    catalog_item_ids: Sequence[str],
    catalog_mean_quantile: float,
    top_k: int = 10,
    ips_gamma: float = 1.0,
) -> CompatMetrics:
    """Aggregate SID / full-catalog metrics (upstream ``sid_eval`` family)."""
    out = CompatMetrics(metrics={"n": float(len(examples))})
    out.sources["n"] = "upstream"
    if not examples:
        return out

    hr1 = [hit_at(ex.target_rank, 1) for ex in examples]
    hrk = [hit_at(ex.target_rank, top_k) for ex in examples]
    ndcg = [
        ndcg_at(ex.target_rank, top_k) if ex.target_rank is not None else 0.0
        for ex in examples
    ]
    _set(out, "hr@1", float(np.mean(hr1)), source="upstream")
    _set(out, f"hr@{top_k}", float(np.mean(hrk)), source="upstream")
    _set(out, f"ndcg@{top_k}", float(np.mean(ndcg)), source="upstream")
    _set(
        out,
        "mrr",
        float(np.mean([mrr_at(ex.target_rank) for ex in examples])),
        source="extended",
    )

    # IPS
    weights, hit_w, ndcg_w = [], [], []
    for ex, h, d in zip(examples, hrk, ndcg, strict=True):
        if ex.target_count is None:
            continue
        w = ips_weight(ex.target_count, ips_gamma)
        weights.append(w)
        hit_w.append(w * h)
        ndcg_w.append(w * d)
    if weights:
        _set(out, f"hr_ips@{top_k}", snips(weights, hit_w), source="upstream")
        _set(out, f"ndcg_ips@{top_k}", snips(weights, ndcg_w), source="upstream")
    else:
        _set(out, f"hr_ips@{top_k}", NOT_APPLICABLE, source="upstream")
        _set(out, f"ndcg_ips@{top_k}", NOT_APPLICABLE, source="upstream")
    out.metrics["ips_gamma"] = float(ips_gamma)
    out.sources["ips_gamma"] = "upstream"

    # Tiers
    tier_hits: dict[str, list[float]] = {"head": [], "mid": [], "tail": []}
    for ex, h in zip(examples, hrk, strict=True):
        if ex.target_pop_quantile is None:
            continue
        tier_hits[popularity_tier(ex.target_pop_quantile)].append(h)
    out.slices["hr_by_tier"] = {
        t: (float(np.mean(v)) if v else NOT_APPLICABLE) for t, v in tier_hits.items()
    }
    out.slices["tier_n"] = {t: len(v) for t, v in tier_hits.items()}

    lifts, gaps = [], []
    for ex in examples:
        if ex.top1_pop_quantile is None:
            continue
        lifts.append(catalog_pop_lift(ex.top1_pop_quantile, catalog_mean_quantile))
        if ex.hist_pop_mean is not None:
            gaps.append(delta_gap(ex.top1_pop_quantile, ex.hist_pop_mean))
    _set(out, "pop_lift@1", float(np.mean(lifts)) if lifts else None, source="upstream")
    _set(out, "delta_gap", float(np.mean(gaps)) if gaps else None, source="upstream")

    # Exposure over full catalog order
    exposure = Counter()
    for ex in examples:
        if ex.exposed_item_ids:
            exposure.update(ex.exposed_item_ids)
    exposure_vec = np.array([exposure.get(i, 0) for i in catalog_item_ids], dtype=np.float64)
    _set(out, "exposure_gini", gini(exposure_vec), source="upstream")
    _set(out, f"coverage@{top_k}", coverage_at(exposure_vec), source="upstream")

    free = [ex.free_gen_valid for ex in examples if ex.free_gen_valid is not None]
    constrained = [
        ex.constrained_valid for ex in examples if ex.constrained_valid is not None
    ]
    _set(
        out,
        "free_gen_valid_rate",
        float(np.mean(free)) if free else None,
        source="upstream",
    )
    _set(
        out,
        "constrained_gen_valid_rate",
        float(np.mean(constrained)) if constrained else None,
        source="extended",
    )

    # Letter-route exclusive
    _set(out, "pop_lift", NOT_APPLICABLE, source="upstream")
    return out


def examples_to_ranked_from_predictions(
    predictions: Iterable[Mapping[str, Any]],
    *,
    features_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[RankedExample]:
    """Convert lab prediction JSONL rows into ``RankedExample`` records."""
    features_by_id = features_by_id or {}
    rows: list[RankedExample] = []
    for pred in predictions:
        eid = str(pred.get("example_id", ""))
        feats = features_by_id.get(eid, {})
        rows.append(
            RankedExample(
                target_rank=pred.get("rank"),
                chosen_index=pred.get("chosen_index"),
                pop_quantiles=feats.get("pop_quantiles") or pred.get("pop_quantiles"),
                top1_pop_quantile=pred.get("top1_pop_quantile"),
                hist_pop_mean=feats.get("history_popularity_mean"),
                target_pop_quantile=feats.get("popularity_quantile"),
                target_count=feats.get("item_popularity"),
                exposed_item_ids=pred.get("exposed_item_ids"),
                free_gen_valid=pred.get("free_gen_valid"),
                constrained_valid=pred.get("constrained_valid"),
            )
        )
    return rows
