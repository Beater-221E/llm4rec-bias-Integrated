"""Plugin reward framework."""

from llm4rec.components.reward.base import BaseReward, RewardBatch, RewardContext
from llm4rec.components.reward.composite import CompositeReward, build_reward_from_config

__all__ = [
    "BaseReward",
    "RewardBatch",
    "RewardContext",
    "CompositeReward",
    "build_reward_from_config",
]
