"""Config-driven composite reward."""

from __future__ import annotations

from typing import Any, Sequence

from llm4rec.components.reward._impl.composer import (
    RewardComposer,
    build_trl_reward_from_config,
)
from llm4rec.components.reward._impl.registry import build_reward, register_reward
from llm4rec.core.exceptions import ConfigurationError


class CompositeReward(RewardComposer):
    """Weighted sum of registered reward plugins."""

    @classmethod
    def from_list(
        cls,
        names: Sequence[str],
        *,
        weights: dict[str, float] | None = None,
        group_size: int | None = None,
        invalid_penalty: float = -0.5,
    ) -> "CompositeReward":
        """Build from a list of reward names (equal weight unless overridden)."""
        if not names:
            raise ConfigurationError("reward list cannot be empty")
        w = {str(n): float((weights or {}).get(str(n), 1.0)) for n in names}
        return cls(w, group_size=group_size, invalid_penalty=invalid_penalty)


def build_reward_from_config(cfg: dict[str, Any] | None) -> RewardComposer:
    """Build rewards from unified ``reward:`` config block.

    Supported shapes::

        reward:
          name: bias_aware
          components: [ndcg, popularity_penalty, position_penalty]
          weights:
            ndcg: 1.0
            popularity_penalty: 0.2
            position_penalty: 0.1

        # or legacy grpo.reward_weights map
        grpo:
          reward_weights:
            exact_match: 1.0
            format_validity: 0.2
    """
    # Ensure plugins are imported
    from llm4rec.components.reward import (  # noqa: F401
        ranking,
        popularity,
        position,
    )
    from llm4rec.components.reward._impl import (  # noqa: F401
        exact_match,
        format_validity,
        popularity_penalty,
        rank_aware,
        sid_prefix,
    )

    cfg = cfg or {}
    reward_cfg = cfg.get("reward") if "reward" in cfg or "grpo" in cfg else cfg
    if reward_cfg is None:
        reward_cfg = {}

    # Nested under full experiment config
    if "grpo" in cfg and "reward" not in cfg:
        return build_trl_reward_from_config(cfg.get("grpo") or {})

    block = cfg.get("reward") if isinstance(cfg.get("reward"), dict) else reward_cfg
    if not isinstance(block, dict):
        block = {}

    components = block.get("components") or block.get("rewards")
    if components:
        weights = dict(block.get("weights") or {})
        for name in components:
            weights.setdefault(str(name), 1.0)
        # Map friendly aliases used in configs
        alias = {
            "accuracy": "exact_match",
            "hr": "hr",
            "ndcg": "ndcg",
            "popularity": "popularity_penalty",
            "popularity_penalty": "popularity_penalty",
            "position": "position_penalty",
            "position_penalty": "position_penalty",
        }
        mapped = {alias.get(k, k): float(v) for k, v in weights.items() if float(v) != 0.0}
        # Prefer exact_match if accuracy listed without exact_match registered path
        if "accuracy" in weights and "exact_match" not in mapped:
            mapped["exact_match"] = float(weights.get("accuracy", 1.0))
            mapped.pop("accuracy", None)
        return CompositeReward(
            mapped,
            group_size=int(block.get("group_size") or 0) or None,
            invalid_penalty=float(block.get("invalid_penalty", -0.5)),
        )

    # Fall back to legacy grpo.reward_weights
    grpo = cfg.get("grpo") or {}
    if isinstance(grpo, dict) and grpo.get("reward_weights"):
        return build_trl_reward_from_config(grpo)

    weights = dict(block.get("weights") or block.get("reward_weights") or {})
    if weights:
        return CompositeReward(weights)

    raise ConfigurationError(
        "No reward components configured. Set reward.components or grpo.reward_weights"
    )


__all__ = [
    "CompositeReward",
    "RewardComposer",
    "build_reward",
    "build_reward_from_config",
    "build_trl_reward_from_config",
    "register_reward",
]
