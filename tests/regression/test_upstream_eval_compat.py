"""Regression: lab metric pure functions match llm4rec-bias definitions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from llm4rec_bias_Integrated.compatibility.llm4rec_bias_eval import (
    NOT_APPLICABLE,
    UPSTREAM_EVAL_VERSION,
    RankedExample,
    aggregate_full_catalog,
    aggregate_letter_route,
    catalog_pop_lift,
    coverage_at,
    delta_gap,
    gini,
    ips_weight,
    letter_pop_lift,
    ndcg_at,
    popularity_tier,
    snips,
)

UPSTREAM_SRC = Path("/home/sheng/proj/llm4rec-bias/src")


@pytest.fixture(scope="module")
def upstream_eval():
    if not (UPSTREAM_SRC / "llm4rec" / "eval.py").is_file():
        pytest.skip("upstream llm4rec-bias not checked out")
    sys.path.insert(0, str(UPSTREAM_SRC))
    import llm4rec.eval as mod  # noqa: WPS433

    return mod


@pytest.fixture(scope="module")
def upstream_sid_eval():
    if not (UPSTREAM_SRC / "llm4rec" / "sid_eval.py").is_file():
        pytest.skip("upstream llm4rec-bias not checked out")
    sys.path.insert(0, str(UPSTREAM_SRC))
    import llm4rec.sid_eval as mod  # noqa: WPS433

    return mod


def test_ndcg_at_matches_upstream(upstream_eval) -> None:
    for rank in range(0, 12):
        for k in (1, 5, 10):
            assert ndcg_at(rank, k) == pytest.approx(upstream_eval.ndcg_at(rank, k))


def test_gini_matches_upstream(upstream_sid_eval) -> None:
    vectors = [
        np.zeros(10),
        np.ones(10),
        np.array([0, 0, 0, 5, 5]),
        np.array([10.0] + [0.0] * 99),
        np.arange(20, dtype=np.float64),
    ]
    for v in vectors:
        assert gini(v) == pytest.approx(upstream_sid_eval.gini(v))


def test_letter_pop_lift_formula() -> None:
    q = [0.1, 0.5, 0.9]
    assert letter_pop_lift(q, 2) == pytest.approx(0.9 - (0.1 + 0.5 + 0.9) / 3)


def test_ips_snips_matches_sid_inline() -> None:
    counts = [100, 1, 10]
    hits = [1.0, 1.0, 0.0]
    gamma = 1.0
    weights = [ips_weight(c, gamma) for c in counts]
    hit_w = [w * h for w, h in zip(weights, hits, strict=True)]
    assert snips(weights, hit_w) == pytest.approx(sum(hit_w) / sum(weights))
    assert ips_weight(100, 1.0) == pytest.approx(0.01)
    assert ips_weight(0, 1.0) == pytest.approx(1.0)


def test_letter_route_marks_sid_metrics_not_applicable() -> None:
    rows = [
        RankedExample(target_rank=0, chosen_index=0, pop_quantiles=[0.2, 0.8]),
        RankedExample(target_rank=3, chosen_index=1, pop_quantiles=[0.1, 0.9]),
    ]
    out = aggregate_letter_route(rows, top_k=[1, 5, 10])
    assert out.metrics["hr@1"] == pytest.approx(0.5)
    assert out.metrics["hr_ips@10"] == NOT_APPLICABLE
    assert out.metrics["exposure_gini"] == NOT_APPLICABLE
    assert out.metrics["free_gen_valid_rate"] == NOT_APPLICABLE
    assert out.sources["hr@1"] == "upstream"
    assert out.sources["mrr"] == "extended"
    assert out.version["compat_version"] == UPSTREAM_EVAL_VERSION["compat_version"]


def test_full_catalog_aggregation_fixture() -> None:
    catalog = [str(i) for i in range(10)]
    rows = [
        RankedExample(
            target_rank=0,
            top1_pop_quantile=0.9,
            hist_pop_mean=0.5,
            target_pop_quantile=0.9,
            target_count=100,
            exposed_item_ids=["0", "1", "2"],
            free_gen_valid=True,
            constrained_valid=True,
        ),
        RankedExample(
            target_rank=5,
            top1_pop_quantile=0.2,
            hist_pop_mean=0.4,
            target_pop_quantile=0.1,
            target_count=1,
            exposed_item_ids=["5", "6"],
            free_gen_valid=False,
            constrained_valid=True,
        ),
    ]
    out = aggregate_full_catalog(
        rows,
        catalog_item_ids=catalog,
        catalog_mean_quantile=0.5,
        top_k=10,
        ips_gamma=1.0,
    )
    assert out.metrics["hr@1"] == pytest.approx(0.5)
    assert out.metrics["hr@10"] == pytest.approx(1.0)
    assert out.metrics["pop_lift@1"] == pytest.approx(
        np.mean([catalog_pop_lift(0.9, 0.5), catalog_pop_lift(0.2, 0.5)])
    )
    assert out.metrics["delta_gap"] == pytest.approx(
        np.mean([delta_gap(0.9, 0.5), delta_gap(0.2, 0.4)])
    )
    assert out.metrics["free_gen_valid_rate"] == pytest.approx(0.5)
    assert out.metrics["constrained_gen_valid_rate"] == pytest.approx(1.0)
    assert out.metrics["pop_lift"] == NOT_APPLICABLE
    assert out.metrics["coverage@10"] == pytest.approx(0.5)
    assert out.metrics["exposure_gini"] == pytest.approx(
        gini(np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0], dtype=np.float64))
    )


def test_coverage_and_tier_helpers() -> None:
    assert popularity_tier(0.0) == "tail"
    assert popularity_tier(0.5) == "mid"
    assert popularity_tier(0.9) == "head"
    assert coverage_at([0, 0, 1, 2]) == pytest.approx(0.5)


def test_letter_aggregate_matches_upstream_loop(upstream_eval) -> None:
    ranks = [0, 2, 0]
    choices = [0, 0, 1]
    quants = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.2, 0.8, 0.5]]
    rows = [
        RankedExample(target_rank=r, chosen_index=c, pop_quantiles=q)
        for r, c, q in zip(ranks, choices, quants, strict=True)
    ]
    out = aggregate_letter_route(rows, top_k=[1, 5])
    # Mirror upstream evaluate() aggregation
    hits = [r == 0 for r in ranks]
    ndcgs = [upstream_eval.ndcg_at(r) for r in ranks]  # default k=5
    lifts = [letter_pop_lift(q, c) for q, c in zip(quants, choices, strict=True)]
    assert out.metrics["hr@1"] == pytest.approx(float(np.mean(hits)))
    assert out.metrics["ndcg@5"] == pytest.approx(float(np.mean(ndcgs)))
    assert out.metrics["pop_lift"] == pytest.approx(float(np.mean(lifts)))
