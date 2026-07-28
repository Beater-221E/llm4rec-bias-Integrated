"""Reward component registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm4rec.core.registry import Registry

REWARD_REGISTRY: Registry[type] = Registry("reward")


def register_reward(name: str) -> Callable[[type], type]:
    return REWARD_REGISTRY.register(name)


def get_reward_class(name: str) -> type:
    # Side-effect imports (legacy + framework plugins)
    from llm4rec.components.reward._impl import (  # noqa: F401
        exact_match,
        format_validity,
        popularity_penalty,
        rank_aware,
        sid_prefix,
    )
    from llm4rec.components.reward import (  # noqa: F401
        ranking,
        popularity,
        position,
    )

    return REWARD_REGISTRY.get(name)


def build_reward(name: str, **kwargs: Any):
    return get_reward_class(name)(**kwargs)
