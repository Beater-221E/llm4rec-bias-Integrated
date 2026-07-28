"""Unit tests for Phase 6 bias probes (no GPU)."""

from __future__ import annotations

import pytest

from llm4rec_bias_Integrated.core.exceptions import ConfigurationError
from llm4rec_bias_Integrated.core.schemas import RecommendationExample
from llm4rec_bias_Integrated.probes.position import _kendall_tau
from llm4rec_bias_Integrated.probes.rebuild import place_target_at_slot, rebuild_example
from llm4rec_bias_Integrated.probes.registry import build_probe, build_probes_from_config
from llm4rec_bias_Integrated.probes.recency import _apply_intervention
import random


def _toy_example() -> RecommendationExample:
    titles = [f"Movie {i}" for i in range(5)]
    quants = [0.1, 0.3, 0.5, 0.7, 0.9]
    ids = [f"i{i}" for i in range(5)]
    hist = ["H1", "H2", "H3"]
    from llm4rec_bias_Integrated.prompts.candidate_choice import build_candidate_choice_messages

    msgs = build_candidate_choice_messages(hist, titles, quants, "neutral")
    return RecommendationExample(
        example_id="toy:0",
        user_id="u0",
        history_item_ids=["h1", "h2", "h3"],
        target_item_id="i2",
        candidates=ids,
        prompt_messages=msgs,
        target_text="C",
        target_index=2,
        features={
            "history_titles": hist,
            "candidate_titles": titles,
            "pop_quantiles": quants,
            "framing": "neutral",
            "candidate_genres": [[] for _ in range(5)],
            "candidate_years": [None] * 5,
        },
    )


def test_rebuild_framing_swap() -> None:
    ex = _toy_example()
    cf = rebuild_example(ex, framing="evaluative", example_id_suffix=":f")
    assert cf.features["framing"] == "evaluative"
    assert "popular hit" in cf.prompt_messages[1]["content"] or "rarely watched" in cf.prompt_messages[1]["content"]
    assert cf.target_index == 2
    assert "Candidates:" in cf.prompt_messages[1]["content"]


def test_place_target_at_all_slots() -> None:
    ex = _toy_example()
    for slot in range(5):
        cf = place_target_at_slot(ex, slot)
        assert cf.target_index == slot
        assert cf.candidates[slot] == "i2"
        assert cf.target_text == "ABCDE"[slot]


def test_history_reverse() -> None:
    titles = ["A", "B", "C"]
    ids = ["a", "b", "c"]
    nt, ni = _apply_intervention(titles, ids, "reverse_history", random.Random(0))
    assert nt == ["C", "B", "A"]
    assert ni == ["c", "b", "a"]


def test_position_spread_math() -> None:
    # Accuracies that vary by slot → positive spread
    acc = [1.0, 0.5, 0.0]
    assert max(acc) - min(acc) == pytest.approx(1.0)
    tau = _kendall_tau([0.0, 1.0, 2.0], [1.0, 0.5, 0.0])
    assert tau is not None
    assert tau < 0  # higher slot → lower accuracy


def test_semantic_prefix_rejected() -> None:
    with pytest.raises(ConfigurationError, match="Phase 7"):
        build_probe("semantic_prefix")


def test_build_probes_from_config() -> None:
    probes = build_probes_from_config(
        {"probes": ["popularity", "position", "history_reversal"]}
    )
    names = [p.name for p in probes]
    assert names == ["popularity", "position", "recency"]
