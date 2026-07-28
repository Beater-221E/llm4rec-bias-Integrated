"""Adapt lab predictions / examples into upstream-compatible metric payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from llm4rec_bias_Integrated.compatibility.llm4rec_bias_eval import (
    UPSTREAM_EVAL_VERSION,
    CompatMetrics,
    RankedExample,
    aggregate_full_catalog,
    aggregate_letter_route,
    examples_to_ranked_from_predictions,
)
from llm4rec_bias_Integrated.core.exceptions import EvaluatorCompatibilityError
from llm4rec_bias_Integrated.core.schemas import EvaluationResult, RecommendationExample

RouteKind = Literal["letter", "full_catalog"]


def examples_to_ranked_letter(
    examples: list[RecommendationExample],
    ranks: list[int],
    chosen: list[int],
) -> list[RankedExample]:
    if not (len(examples) == len(ranks) == len(chosen)):
        raise EvaluatorCompatibilityError(
            "examples/ranks/chosen length mismatch for letter-route adapter"
        )
    rows: list[RankedExample] = []
    for ex, rank, ch in zip(examples, ranks, chosen, strict=True):
        quants = ex.features.get("pop_quantiles")
        rows.append(
            RankedExample(
                target_rank=int(rank),
                chosen_index=int(ch),
                pop_quantiles=list(quants) if quants is not None else None,
            )
        )
    return rows


def to_evaluation_result(
    compat: CompatMetrics,
    *,
    split: str,
    predictions_path: Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> EvaluationResult:
    meta = {
        "split": split,
        "metric_sources": dict(compat.sources),
        "evaluator_version": dict(compat.version or UPSTREAM_EVAL_VERSION),
        **(extra_metadata or {}),
    }
    # Keep float metrics in metrics; stringify N/A stays as str (schema allows Any via cast)
    return EvaluationResult(
        metrics=dict(compat.metrics),  # type: ignore[arg-type]
        slices={k: dict(v) for k, v in compat.slices.items()},  # type: ignore[arg-type]
        predictions_path=predictions_path,
        metadata=meta,
    )


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    *,
    route: RouteKind,
    split: str = "test",
    top_k: list[int] | None = None,
    catalog_item_ids: list[str] | None = None,
    catalog_mean_quantile: float = 0.5,
    ips_gamma: float = 1.0,
    features_by_id: dict[str, dict[str, Any]] | None = None,
    predictions_path: Path | None = None,
) -> EvaluationResult:
    """Score precomputed prediction rows with the upstream-compatible aggregator."""
    if not predictions:
        raise EvaluatorCompatibilityError("No predictions provided to evaluate_predictions")
    ranked = examples_to_ranked_from_predictions(predictions, features_by_id=features_by_id)
    if route == "letter":
        compat = aggregate_letter_route(ranked, top_k=top_k or [1, 5, 10])
    else:
        if not catalog_item_ids:
            raise EvaluatorCompatibilityError(
                "full_catalog route requires catalog_item_ids"
            )
        k = max(top_k or [10])
        compat = aggregate_full_catalog(
            ranked,
            catalog_item_ids=catalog_item_ids,
            catalog_mean_quantile=catalog_mean_quantile,
            top_k=k,
            ips_gamma=ips_gamma,
        )
    return to_evaluation_result(compat, split=split, predictions_path=predictions_path)
