"""Evaluator / reward plugin tests."""

from __future__ import annotations

from llm4rec.components.evaluation.bias import BiasMetrics
from llm4rec.components.evaluation.generation import GenerationMetrics
from llm4rec.components.evaluation.ranking import RankingMetrics, hit_at_k, mrr, ndcg_at_k
from llm4rec.components.reward.composite import CompositeReward, build_reward_from_config
from llm4rec.components.reward._impl.registry import REWARD_REGISTRY, get_reward_class


def test_ranking_metrics_aggregate():
    metrics = RankingMetrics(top_k=[1, 5, 10]).evaluate([1, 3, None, 12])
    assert metrics["HR@1"] == 0.25
    assert metrics["HR@5"] == 0.5
    assert 0.0 < metrics["MRR"] < 1.0
    assert hit_at_k(1, 1) == 1.0
    assert ndcg_at_k(2, 5) > 0
    assert mrr(None) == 0.0


def test_generation_metrics():
    v = GenerationMetrics.sid_validity([True, True, False])
    assert abs(v["sid_validity"] - 2 / 3) < 1e-9
    c = GenerationMetrics.semantic_collision([(0, 1), (0, 1), (2, 3)])
    assert c["semantic_collision_rate"] > 0
    a = GenerationMetrics.generation_accuracy([True, False])
    assert a["generation_accuracy"] == 0.5


def test_bias_metrics_empty_report():
    report = BiasMetrics.empty_report()
    assert "popularity_bias" in report
    assert "position_bias" in report


def test_reward_plugins_registered():
    for name in ("exact_match", "hr", "ndcg", "popularity_penalty", "position_penalty", "accuracy"):
        cls = get_reward_class(name)
        assert cls is not None
        assert REWARD_REGISTRY.contains(name)


def test_composite_reward_from_config_list():
    composer = build_reward_from_config(
        {
            "reward": {
                "components": ["ndcg", "popularity_penalty", "position_penalty"],
                "weights": {
                    "ndcg": 1.0,
                    "popularity_penalty": 0.2,
                    "position_penalty": 0.1,
                },
            }
        }
    )
    assert isinstance(composer, CompositeReward)
    assert set(composer.weights) >= {"ndcg", "popularity_penalty", "position_penalty"}
