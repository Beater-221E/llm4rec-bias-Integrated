"""Rewards package."""

from llm4rec_bias_Integrated.rewards.composer import RewardComposer, build_trl_reward_from_config
from llm4rec_bias_Integrated.rewards.registry import build_reward, register_reward

__all__ = [
    "RewardComposer",
    "build_reward",
    "build_trl_reward_from_config",
    "register_reward",
]
