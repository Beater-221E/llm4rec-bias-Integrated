"""Compose weighted reward components into a TRL-compatible callable."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch

from llm4rec.core.exceptions import ConfigurationError
from llm4rec.components.prompts.candidate_choice import parse_choice
from llm4rec.components.reward._impl.base import RewardBatch, RewardContext, completion_text
from llm4rec.components.reward._impl.registry import build_reward


class RewardComposer:
    """Weighted sum of registered reward components.

    Also emits upstream-compatible shortcut telemetry via ``log_metric``.
    """

    def __init__(
        self,
        weights: dict[str, float],
        *,
        group_size: int | None = None,
        invalid_penalty: float = -0.5,
    ) -> None:
        active = {k: float(v) for k, v in weights.items() if float(v) != 0.0}
        if not active:
            raise ConfigurationError("grpo.reward_weights cannot be all zeros")
        self.weights = active
        self.components = {}
        for name in active:
            kwargs: dict[str, Any] = {}
            if name == "rank_aware" and group_size is not None:
                kwargs["group_size"] = group_size
            if name == "format_validity":
                kwargs["invalid_penalty"] = invalid_penalty
            self.components[name] = build_reward(name, **kwargs)

    def __call__(
        self,
        prompts: Sequence[Any],
        completions: Sequence[Any],
        target: Sequence[int] | None = None,
        pop_quantiles: Sequence[Sequence[float]] | None = None,
        log_metric: Callable[[str, float], None] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        outputs = [completion_text(c) for c in completions]
        batch = RewardBatch(
            prompts=prompts,
            completions=completions,
            targets=list(target) if target is not None else None,
            pop_quantiles=[list(q) for q in pop_quantiles] if pop_quantiles is not None else None,
            extras=dict(kwargs),
        )
        context = RewardContext(log_metric=log_metric, stage="grpo")

        total = torch.zeros(len(outputs), dtype=torch.float32)
        telemetry: dict[str, float] = {}
        component_means: dict[str, float] = {}

        for name, weight in self.weights.items():
            out = self.components[name](batch, outputs, context)
            total = total + weight * out.total.cpu().float()
            component_means[name] = float(out.total.float().mean())
            telemetry.update(out.telemetry)

        # Shared shortcut telemetry (upstream letter route)
        invalid = 0
        positions: list[float] = []
        lifts: list[float] = []
        if batch.pop_quantiles is not None and batch.targets is not None:
            for text, tgt, quants in zip(outputs, batch.targets, batch.pop_quantiles, strict=True):
                n = len(quants)
                choice = parse_choice(text, n)
                if choice is None:
                    invalid += 1
                    continue
                positions.append(choice / max(n - 1, 1))
                lifts.append(float(quants[choice]) - float(np.mean(quants)))

        if log_metric is not None:
            # Always emit a fixed key set so TRL's per-key accelerator.gather
            # stays aligned across ranks (conditional keys deadlock DDP).
            log_metric("reward/total", float(total.mean()))
            for name, mean in sorted(component_means.items()):
                log_metric(f"reward/{name}", mean)
            log_metric("shortcut/invalid_rate", invalid / max(len(outputs), 1))
            log_metric(
                "shortcut/chosen_pos_mean",
                float(np.mean(positions)) if positions else 0.0,
            )
            log_metric(
                "shortcut/pop_lift", float(np.mean(lifts)) if lifts else 0.0
            )
            log_metric(
                "shortcut/exact_hit_rate",
                float(telemetry.get("exact_hit_rate", 0.0)),
            )

        return [float(x) for x in total.tolist()]


def build_trl_reward_from_config(grpo_cfg: dict[str, Any]) -> RewardComposer:
    weights = dict(grpo_cfg.get("reward_weights") or {})
    # Defaults aligned with letter-route upstream if unspecified
    if not weights:
        weights = {"exact_match": 1.0, "format_validity": 0.2}
    return RewardComposer(
        weights,
        group_size=int(grpo_cfg.get("num_generations") or 0) or None,
        invalid_penalty=float(grpo_cfg.get("invalid_penalty", -0.5)),
    )
