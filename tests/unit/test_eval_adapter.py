"""Unit tests for evaluate_predictions adapter."""

from __future__ import annotations

from llm4rec_bias_Integrated.compatibility.llm4rec_bias_eval import NOT_APPLICABLE
from llm4rec_bias_Integrated.evaluation.adapter import evaluate_predictions


def test_evaluate_predictions_letter_route() -> None:
    preds = [
        {
            "example_id": "a",
            "rank": 0,
            "chosen_index": 0,
            "pop_quantiles": [0.1, 0.9],
        },
        {
            "example_id": "b",
            "rank": 1,
            "chosen_index": 1,
            "pop_quantiles": [0.2, 0.8],
        },
    ]
    result = evaluate_predictions(preds, route="letter", split="test")
    assert result.metrics["hr@1"] == 0.5
    assert result.metrics["hr_ips@10"] == NOT_APPLICABLE
    assert result.metadata["metric_sources"]["hr@1"] == "upstream"
    assert "evaluator_version" in result.metadata
