"""物品目录：流行度计数、分位、head/mid/tail 分层。

三条路线共用同一份 catalog —— bias 指标要可比，流行度的定义就必须完全一致。
catalog 从预处理产物 ``item_meta.json`` + ``popularity.json`` 构建，
统计只用 **train split** 的交互，避免把 test 的信息泄进流行度先验。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from llm4rec.core.exceptions import MissingArtifactError


@dataclass
class ItemCatalog:
    """全库物品 + 流行度统计。"""

    item_ids: list[str]
    counts: dict[str, int]
    titles: dict[str, str]

    # ------------------------------------------------------------------ 构建
    @classmethod
    def from_processed(cls, processed_dir: Path) -> ItemCatalog:
        meta_path = Path(processed_dir) / "item_meta.json"
        pop_path = Path(processed_dir) / "popularity.json"
        for path in (meta_path, pop_path):
            if not path.is_file():
                raise MissingArtifactError(
                    f"缺少 {path}，先跑 prepare.sh 的 data 步骤"
                )
        with meta_path.open(encoding="utf-8") as fh:
            meta = json.load(fh)
        with pop_path.open(encoding="utf-8") as fh:
            pop = json.load(fh)

        counts = _extract_counts(pop)
        item_ids = sorted(meta.keys())
        titles = {str(k): str((v or {}).get("title") or "") for k, v in meta.items()}
        return cls(
            item_ids=item_ids,
            counts={i: int(counts.get(i, 0)) for i in item_ids},
            titles=titles,
        )

    # -------------------------------------------------------------- 流行度
    @cached_property
    def quantiles(self) -> dict[str, float]:
        """每个物品的流行度分位（0=最冷门，1=最热门）。

        用 ``rank / (n-1)``，同 count 的物品取平均秩，保证并列物品分位相同。
        """
        n = len(self.item_ids)
        if n == 0:
            return {}
        if n == 1:
            return {self.item_ids[0]: 0.5}

        vals = np.array([self.counts[i] for i in self.item_ids], dtype=np.float64)
        order = np.argsort(vals, kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)

        # 并列取平均秩
        for value in np.unique(vals):
            mask = vals == value
            if mask.sum() > 1:
                ranks[mask] = ranks[mask].mean()

        return {
            item: float(rank / (n - 1))
            for item, rank in zip(self.item_ids, ranks, strict=True)
        }

    @cached_property
    def mean_quantile(self) -> float:
        """全库平均分位。``pop_lift`` 就是相对这个基线算的（理论上 ≈0.5）。"""
        q = self.quantiles
        return float(np.mean(list(q.values()))) if q else 0.5

    def quantile(self, item_id: str) -> float:
        return self.quantiles.get(str(item_id), 0.5)

    def count(self, item_id: str) -> int:
        return int(self.counts.get(str(item_id), 0))

    def history_pop_mean(self, history: Sequence[str]) -> float:
        """用户历史的平均流行度，``delta_gap`` 的个性化基线。"""
        vals = [self.quantile(i) for i in history if str(i) in self.counts]
        return float(np.mean(vals)) if vals else self.mean_quantile

    # ---------------------------------------------------------------- 分层
    def tier(self, item_id: str, thresholds: Mapping[str, float] | None = None) -> str:
        """head / mid / tail。

        默认按三等分，与上游 llm4rec-bias 的 ``popularity_tier`` 一致
        （这样我们的数字能直接和大哥框架的对上）。
        """
        q = self.quantile(item_id)
        if thresholds:
            tail = float(thresholds.get("tail", 1.0 / 3.0))
            head = float(thresholds.get("head", 2.0 / 3.0))
        else:
            tail, head = 1.0 / 3.0, 2.0 / 3.0
        if q < tail:
            return "tail"
        if q < head:
            return "mid"
        return "head"

    def __len__(self) -> int:
        return len(self.item_ids)


def _extract_counts(pop: dict) -> dict[str, int]:
    """兼容 popularity.json 的几种写法。"""
    if isinstance(pop.get("counts"), dict):
        return {str(k): int(v) for k, v in pop["counts"].items()}
    out: dict[str, int] = {}
    for key, value in pop.items():
        if isinstance(value, dict):
            out[str(key)] = int(value.get("count", 0))
        elif isinstance(value, (int, float)):
            out[str(key)] = int(value)
    return out
