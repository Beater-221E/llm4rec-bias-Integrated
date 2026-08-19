"""统一 bias 评测 —— 三条路线共用的唯一实现。

★ 核心约定：所有 bias 指标都算在 **最终的 ranked item list** 上。

三条路线的 LLM 输出完全不同：

    MiniOneRec : SID token 序列  → 约束 beam 解码        → ranked list
    Rec-R1     : 检索 query      → BM25 检索              → ranked list
    DPO4Rec    : 推理文本        → adaptor + reranker      → ranked list

只有 ranked list 这一层是可比的。每条路线各自实现一个 Decoder，把自己的输出
归一成 ``RankedResult``，之后的 bias 计算就完全共用这里的代码 —— 这是让
"backbone 一样、数据集一样、bias 指标可比"真正成立的唯一接缝。

指标的数学定义全部复用 ``llm4rec.compatibility.llm4rec_bias_eval`` 里从上游
llm4rec-bias 移植的纯函数，保证和大哥框架的数字位级一致。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from llm4rec.compatibility.llm4rec_bias_eval import (
    catalog_pop_lift,
    coverage_at,
    delta_gap,
    gini,
    hit_at,
    ips_weight,
    mrr_at,
    ndcg_at,
    snips,
)
from llm4rec.eval.catalog import ItemCatalog


@dataclass
class RankedResult:
    """一个用户一次预测的归一化结果。三条路线的 Decoder 都产出这个。"""

    user_id: str
    ranked_items: list[str]
    target_item: str
    history: list[str] = field(default_factory=list)
    # 该路线的原始输出是否合法（SID 能解析 / query 非空 / 推理文本可用）
    valid: bool = True

    @property
    def target_rank(self) -> int | None:
        """目标物品在结果里的 0-based 位置；没命中返回 None。"""
        try:
            return self.ranked_items.index(str(self.target_item))
        except ValueError:
            return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "ranked_items": list(self.ranked_items),
            "target_item": self.target_item,
            "history": list(self.history),
            "valid": bool(self.valid),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RankedResult:
        return cls(
            user_id=str(raw.get("user_id") or ""),
            ranked_items=[str(x) for x in (raw.get("ranked_items") or [])],
            target_item=str(raw.get("target_item") or ""),
            history=[str(x) for x in (raw.get("history") or [])],
            valid=bool(raw.get("valid", True)),
        )


def compute_bias_metrics(
    results: Sequence[RankedResult],
    catalog: ItemCatalog,
    *,
    top_k: int = 10,
    ips_gamma: float = 1.0,
    tier_thresholds: Mapping[str, float] | None = None,
    enabled: Sequence[str] | None = None,
) -> dict[str, Any]:
    """在 ranked list 上算全套 bias 指标。

    ``enabled`` 是 ``configs/bias/*.yaml`` 里的 ``metrics`` 列表；
    传 None 表示全算。返回值是扁平的 ``{name: float}``，可以直接喂给 wandb。
    """
    if not results:
        return {"n": 0.0}

    want = set(enabled) if enabled is not None else None

    def on(name: str) -> bool:
        return want is None or name in want

    out: dict[str, Any] = {"n": float(len(results))}

    ranks = [r.target_rank for r in results]
    hits_k = [hit_at(rank, top_k) for rank in ranks]
    ndcgs_k = [ndcg_at(rank, top_k) if rank is not None else 0.0 for rank in ranks]

    # —— 基础准确率（bias 要和它一起看：RL 通常准确率涨、bias 也涨）——
    out["hr@1"] = float(np.mean([hit_at(r, 1) for r in ranks]))
    out[f"hr@{top_k}"] = float(np.mean(hits_k))
    out[f"ndcg@{top_k}"] = float(np.mean(ndcgs_k))
    out["mrr"] = float(np.mean([mrr_at(r) for r in ranks]))
    out["valid_rate"] = float(np.mean([float(r.valid) for r in results]))

    # ------------------------------------------------------------ 流行度偏置
    if on("pop_lift"):
        lifts_1, lifts_k = [], []
        for res in results:
            if not res.ranked_items:
                continue
            lifts_1.append(
                catalog_pop_lift(catalog.quantile(res.ranked_items[0]), catalog.mean_quantile)
            )
            topk = res.ranked_items[:top_k]
            if topk:
                mean_q = float(np.mean([catalog.quantile(i) for i in topk]))
                lifts_k.append(catalog_pop_lift(mean_q, catalog.mean_quantile))
        out["pop_lift@1"] = float(np.mean(lifts_1)) if lifts_1 else 0.0
        out[f"pop_lift@{top_k}"] = float(np.mean(lifts_k)) if lifts_k else 0.0

    if on("delta_gap"):
        # ΔGAP：推荐的流行度 − 该用户自己历史的流行度。
        # 这个比 pop_lift 更能说明问题：它剔除了"用户本来就爱看热门"的成分。
        gaps = [
            delta_gap(catalog.quantile(res.ranked_items[0]), catalog.history_pop_mean(res.history))
            for res in results
            if res.ranked_items
        ]
        out["delta_gap"] = float(np.mean(gaps)) if gaps else 0.0

    # ------------------------------------------------------------ 曝光集中度
    exposure = Counter()
    for res in results:
        exposure.update(res.ranked_items[:top_k])
    exposure_vec = np.array(
        [exposure.get(i, 0) for i in catalog.item_ids], dtype=np.float64
    )

    if on("exposure_gini"):
        out["exposure_gini"] = gini(exposure_vec)
    if on("exposure_entropy"):
        out["exposure_entropy"] = _normalized_entropy(exposure_vec)
    if on("coverage"):
        out[f"coverage@{top_k}"] = coverage_at(exposure_vec)

    # -------------------------------------------------- 分层准确率（长尾代价）
    if on("tier_hr"):
        # RL 常见的"作弊"方式：放弃长尾、只押热门，总分涨但 tail 崩。
        # 分开看 head/mid/tail 才抓得住。
        tiers: dict[str, list[float]] = {"head": [], "mid": [], "tail": []}
        for res, hit in zip(results, hits_k, strict=True):
            tiers[catalog.tier(res.target_item, tier_thresholds)].append(hit)
        for name, vals in tiers.items():
            out[f"hr@{top_k}_{name}"] = float(np.mean(vals)) if vals else 0.0
            out[f"n_{name}"] = float(len(vals))
        head, tail = out[f"hr@{top_k}_head"], out[f"hr@{top_k}_tail"]
        out["tier_gap"] = float(head - tail)

    # ------------------------------------------------- IPS 去偏后的准确率
    if on("ips"):
        # 用 1/count^gamma 给长尾目标加权。如果去偏后准确率不涨甚至跌，
        # 说明模型的提升主要来自"猜热门"。
        weights, hit_w, ndcg_w = [], [], []
        for res, hit, ndcg in zip(results, hits_k, ndcgs_k, strict=True):
            w = ips_weight(catalog.count(res.target_item), ips_gamma)
            weights.append(w)
            hit_w.append(w * hit)
            ndcg_w.append(w * ndcg)
        out[f"hr_ips@{top_k}"] = snips(weights, hit_w) or 0.0
        out[f"ndcg_ips@{top_k}"] = snips(weights, ndcg_w) or 0.0

    # ------------------------------------------------------------ shortcut
    if on("history_copy_rate"):
        # 直接把用户历史里的物品吐回去 —— 生成式推荐里最常见的 shortcut。
        rates = []
        for res in results:
            topk = res.ranked_items[:top_k]
            if not topk:
                continue
            hist = set(map(str, res.history))
            rates.append(sum(1 for i in topk if str(i) in hist) / len(topk))
        out["history_copy_rate"] = float(np.mean(rates)) if rates else 0.0

    if on("top1_concentration"):
        # 所有用户的 top1 落在多少个不同物品上。趋近 0 = 个性化坍塌成一个全局热门榜。
        tops = [res.ranked_items[0] for res in results if res.ranked_items]
        out["top1_distinct"] = float(len(set(tops)))
        out["top1_concentration"] = (
            float(1.0 - len(set(tops)) / len(tops)) if tops else 0.0
        )

    return out


def _normalized_entropy(counts: np.ndarray) -> float:
    """曝光分布的归一化熵，1 = 完全均匀，0 = 全压在一个物品上。"""
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    nonzero = counts[counts > 0] / total
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    max_entropy = float(np.log(len(counts))) if len(counts) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


def bias_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float]:
    """两次评测之间的 bias 变化量（典型用法：SFT 基线 → RL 结束）。

    这直接就是"RL 是否放大了 bias"的核心证据表。
    """
    out: dict[str, float] = {}
    for key, new in after.items():
        old = before.get(key)
        if isinstance(new, (int, float)) and isinstance(old, (int, float)):
            out[f"delta/{key}"] = float(new) - float(old)
    return out
