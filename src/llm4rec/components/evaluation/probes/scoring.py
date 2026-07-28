"""Shared letter-choice scoring for probes."""

from __future__ import annotations

from typing import Any

import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llm4rec.compatibility.llm4rec_bias_eval import letter_pop_lift, popularity_tier
from llm4rec.core.schemas import RecommendationExample
from llm4rec.components.evaluation._impl.ranking import score_letters


def choose_index(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: Any,
    ex: RecommendationExample,
) -> int:
    n = len(ex.candidates or ex.features.get("candidate_titles") or [])
    if n <= 0:
        raise ValueError(f"no candidates on {ex.example_id}")
    scores = score_letters(tokenizer, model, device, ex.prompt_messages, n)
    return int(np.argmax(scores))


def score_example(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: Any,
    ex: RecommendationExample,
) -> dict[str, Any]:
    """Return choice, hit@1, pop_lift, tier for one example."""
    choice = choose_index(tokenizer, model, device, ex)
    target = int(ex.target_index) if ex.target_index is not None else -1
    quants = [float(q) for q in (ex.features.get("pop_quantiles") or [])]
    lift = letter_pop_lift(quants, choice) if quants else 0.0
    tier = popularity_tier(quants[choice]) if quants else "mid"
    return {
        "choice": choice,
        "hit": float(choice == target),
        "pop_lift": float(lift),
        "tier": tier,
        "target": target,
        "n_candidates": len(quants) if quants else 0,
    }
