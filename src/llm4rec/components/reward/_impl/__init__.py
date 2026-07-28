"""Rewards package."""

from llm4rec.components.reward._impl.composer import RewardComposer, build_trl_reward_from_config
from llm4rec.components.reward._impl.registry import build_reward, register_reward

__all__ = [
    "RewardComposer",
    "build_reward",
    "build_trl_reward_from_config",
    "register_reward",
]
