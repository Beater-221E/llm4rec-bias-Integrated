"""Popularity bias penalty reward (facade; registered in _impl)."""

from __future__ import annotations

from llm4rec.components.reward._impl.popularity_penalty import PopularityPenaltyReward
from llm4rec.components.reward._impl.registry import REWARD_REGISTRY, register_reward

# Alias registration without clobbering the canonical name
if not REWARD_REGISTRY.contains("popularity"):
    register_reward("popularity")(PopularityPenaltyReward)

PopularityPenalty = PopularityPenaltyReward

__all__ = ["PopularityPenalty", "PopularityPenaltyReward"]
