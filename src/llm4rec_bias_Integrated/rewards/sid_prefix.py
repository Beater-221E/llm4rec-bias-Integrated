"""SID prefix-credit reward (exact / prefix / invalid)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from llm4rec_bias_Integrated.rewards.base import RewardBatch, RewardContext, completion_text
from llm4rec_bias_Integrated.rewards.registry import register_reward
from llm4rec_bias_Integrated.semantic_ids.table import SidTable


@register_reward("sid_prefix")
class SidPrefixReward:
    name = "sid_prefix"

    def __init__(
        self,
        *,
        sid_table_path: str,
        item_meta_path: str,
        prefix_credit: float = 0.1,
        invalid_penalty: float = -0.5,
        **_: Any,
    ) -> None:
        self.table = SidTable(sid_table_path)
        with open(item_meta_path, encoding="utf-8") as f:
            self.meta = {str(k): v for k, v in json.load(f).items()}
        self.prefix_credit = float(prefix_credit)
        self.invalid_penalty = float(invalid_penalty)
        self.catalog_pop_mean = float(
            np.mean([float(m["pop_quantile"]) for m in self.meta.values()])
        )

    def __call__(
        self,
        batch: RewardBatch,
        outputs: Sequence[str],
        context: RewardContext,
    ):
        import torch

        from llm4rec_bias_Integrated.core.schemas import RewardOutput

        targets = batch.targets
        if targets is None:
            # TRL may pass target_item via extras
            targets = batch.extras.get("target_item")
        if targets is None:
            raise ValueError("sid_prefix requires target_item / targets")

        target_ids = [str(t) for t in targets]
        items = [self.table.parse(text) for text in outputs]
        rewards: list[float] = []
        depths: list[float] = []
        for item, tgt in zip(items, target_ids, strict=True):
            if item is None:
                rewards.append(self.invalid_penalty)
                continue
            if item == tgt:
                rewards.append(1.0)
                depths.append(float(self.table.levels))
                continue
            depth = 0
            for a, b in zip(self.table.codes[item], self.table.codes[tgt], strict=False):
                if a != b:
                    break
                depth += 1
            rewards.append(self.prefix_credit * depth)
            depths.append(float(depth))

        if context.log_metric is not None:
            # Always log the same metric keys on every rank. TRL flushes
            # log_metric via accelerator.gather per key; conditional keys
            # deadlock multi-GPU when ranks diverge (e.g. one rank parses a
            # valid SID while others are all-invalid).
            invalid = sum(i is None for i in items)
            context.log_metric("shortcut/invalid_rate", invalid / max(len(items), 1))
            lifts = [
                float(self.meta[i]["pop_quantile"]) - self.catalog_pop_mean
                for i in items
                if i is not None and i in self.meta
            ]
            context.log_metric(
                "shortcut/pop_lift", float(np.mean(lifts)) if lifts else 0.0
            )
            context.log_metric(
                "shortcut/prefix_depth", float(np.mean(depths)) if depths else 0.0
            )

        total = torch.tensor(rewards, dtype=torch.float32)
        return RewardOutput(total=total, components={"sid_prefix": total}, telemetry={})


def build_trl_sid_prefix_reward(
    *,
    sid_table_path: str,
    item_meta_path: str,
    prefix_credit: float = 0.1,
    invalid_penalty: float = -0.5,
):
    """TRL-compatible callable (prompts, completions, target_item=...)."""
    component = SidPrefixReward(
        sid_table_path=sid_table_path,
        item_meta_path=item_meta_path,
        prefix_credit=prefix_credit,
        invalid_penalty=invalid_penalty,
    )

    def reward_fn(prompts, completions, target_item=None, log_metric=None, **kwargs):
        texts = [completion_text(c) for c in completions]
        batch = RewardBatch(
            prompts=prompts,
            completions=completions,
            targets=list(target_item) if target_item is not None else None,
            extras=dict(kwargs),
        )
        context = RewardContext(log_metric=log_metric, stage="grpo")
        out = component(batch, texts, context)
        return [float(x) for x in out.total.tolist()]

    reward_fn.__name__ = "SidPrefixReward"
    return reward_fn
