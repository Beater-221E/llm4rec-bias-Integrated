"""Unit tests for reward composition and hacking analysis."""

from __future__ import annotations

import pytest

from llm4rec_bias_Integrated.core.exceptions import ConfigurationError
from llm4rec_bias_Integrated.evaluation.hacking import analyze_reward_hacking, hacking_gap, pearson
from llm4rec_bias_Integrated.prompts.candidate_choice import parse_choice
from llm4rec_bias_Integrated.rewards.composer import RewardComposer


def test_reward_composer_exact_and_format() -> None:
    composer = RewardComposer(
        {"exact_match": 1.0, "format_validity": 0.2},
        invalid_penalty=-0.5,
    )
    rewards = composer(
        prompts=[{}] * 3,
        completions=["A", "B", "Based on history"],
        target=[0, 0, 0],
        pop_quantiles=[[0.1, 0.9]] * 3,
    )
    # A exact → 1.0 + 0.2*1.0 = 1.2
    # B miss  → 0.0 + 0.2*1.0 = 0.2
    # invalid → 0.0 + 0.2*(-0.5) = -0.1
    assert rewards[0] == pytest.approx(1.2)
    assert rewards[1] == pytest.approx(0.2)
    assert rewards[2] == pytest.approx(-0.1)


def test_zero_weights_rejected() -> None:
    with pytest.raises(ConfigurationError):
        RewardComposer({"exact_match": 0.0, "format_validity": 0.0})


def test_parser_shared() -> None:
    assert parse_choice("A", 10) == 0
    assert parse_choice("nope", 10) is None


def test_hacking_gap_relative() -> None:
    gap = hacking_gap([0.0, 1.0], [0.1, 0.05], method="relative")
    assert gap["delta_reward_raw"] == pytest.approx(1.0)
    assert gap["delta_heldout_raw"] == pytest.approx(-0.05)
    # floor |x0| at 1.0 → Δreward_norm=1.0, Δheldout_norm=-0.05
    assert gap["hacking_gap"] == pytest.approx(1.0 - (-0.05))


def test_analyze_reward_hacking() -> None:
    report = analyze_reward_hacking(
        [
            {"train/reward": 0.0, "eval/hr@10": 0.4},
            {"train/reward": 0.5, "eval/hr@10": 0.3},
        ]
    )
    assert report["n_checkpoints"] == 2
    assert "relative" in report["gaps"]
    assert pearson([0.0, 0.5], [0.4, 0.3]) is not None
