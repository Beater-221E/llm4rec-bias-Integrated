"""Evaluation runner — load a run dir / checkpoint and score a split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm4rec.core.exceptions import MissingArtifactError
from llm4rec.core.reproducibility import write_json
from llm4rec.core.schemas import EvaluationResult
from llm4rec.components.evaluation._impl.adapter import evaluate_predictions
from llm4rec.components.model._impl.base import require_cuda


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise MissingArtifactError(f"Predictions not found: {path}")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_run_predictions(
    run_dir: Path,
    *,
    split: str = "test",
    route: str = "letter",
    top_k: list[int] | None = None,
) -> EvaluationResult:
    """Re-score saved predictions with the compatibility aggregator (CPU-safe math)."""
    pred_path = run_dir / "predictions" / f"predictions_{split}.jsonl"
    rows = load_prediction_rows(pred_path)
    result = evaluate_predictions(
        rows,
        route="letter" if route == "letter" else "full_catalog",
        split=split,
        top_k=top_k or [1, 5, 10],
        predictions_path=pred_path,
    )
    out = run_dir / "eval" / f"{split}_compat_metrics.json"
    write_json(
        out,
        {
            "metrics": result.metrics,
            "slices": result.slices,
            "metadata": result.metadata,
        },
    )
    return result


def evaluate_checkpoint_on_examples(
    *,
    model_cfg: dict[str, Any],
    examples: list,
    adapter_path: str | None,
    sft_adapter_path: str | None = None,
    top_k: list[int] | None = None,
    predictions_dir: Path | None = None,
    split: str = "test",
) -> EvaluationResult:
    """GPU path: score examples with letter log-probs."""
    require_cuda()
    from llm4rec.components.evaluation._impl.ranking import CandidateLogProbEvaluator
    from llm4rec.components.model._impl.loader import load_model_bundle

    tok, model, _ = load_model_bundle(
        model_cfg,
        peft_cfg=None,
        adapter_path=adapter_path,
        sft_adapter_path=sft_adapter_path,
        for_training=False,
        local_rank=0,
    )
    evaluator = CandidateLogProbEvaluator(
        tok,
        model,
        top_k=top_k or [1, 5, 10],
        predictions_dir=predictions_dir,
    )
    return evaluator.evaluate(model, examples, split=split)
