"""Config composition smoke for new reward/evaluation slots."""

from __future__ import annotations

from llm4rec.core.config import load_config, validate_config


def test_compose_reward_and_evaluation_slots():
    cfg = load_config(
        [
            "experiment=smoke_test",
            "reward=bias_aware",
            "evaluation=ranking",
            "model=qwen25_0_5b",
        ]
    )
    data = validate_config(cfg)
    assert data["reward"]["name"] == "bias_aware"
    assert "ndcg" in data["reward"]["components"]
    assert data["evaluation"]["name"] == "ranking"
    assert data["model"]["checkpoint"]


def test_compose_dataset_alias_ml1m():
    cfg = load_config(["experiment=smoke_test", "dataset=ml1m"])
    data = validate_config(cfg)
    assert data["dataset"]["name"] == "movielens_1m"
