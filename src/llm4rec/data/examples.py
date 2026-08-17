"""三条路线的样本构建 —— 共用同一份交互数据和同一份物品文本。

统一的样本 schema（不管哪条路线）：

    user_id      用户
    history      历史 item id 列表（时间序）
    target_item  目标 item id
    prompt       chat messages 列表
    answer       SFT 的监督目标（RL 阶段用不到）
    split        train / val / test

这样 bias evaluator 拿到任何一条路线的样本都能算 —— 因为 ``history`` /
``target_item`` 的语义完全一致，``delta_gap`` 之类依赖历史的指标才可比。
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from llm4rec.core.exceptions import ConfigurationError

# ---------------------------------------------------------------- 通用 prompt

SYSTEM_SID = (
    "You are a generative recommender. Given a user's interaction history, "
    "predict the next item as Semantic ID tokens."
)

SYSTEM_QUERY = (
    "You are a product search assistant. Given a user's purchase history, "
    "write a search query that retrieves the item the user will buy next.\n"
    "First reason inside <think></think>, then output the query inside "
    '<answer></answer> as JSON: {"query": "..."}'
)

SYSTEM_REASONING = (
    "You are a helpful assistant capable of summarizing and extracting user "
    "preferences using natural language."
)


def _titles(history: Sequence[str], meta: dict[str, dict[str, Any]], limit: int = 80) -> list[str]:
    out = []
    for item in history:
        title = str((meta.get(str(item)) or {}).get("title") or "").strip()
        out.append(title[:limit] if title else str(item))
    return out


# ------------------------------------------------------------ 路线 1：SID 生成


def sid_seqrec_example(
    *,
    user_id: str,
    history: Sequence[str],
    target: str,
    sid_table: Any,
    meta: dict[str, dict[str, Any]],
    with_titles: bool = True,
    category_prompt: str = "items",
) -> dict[str, Any] | None:
    """序列推荐主任务：历史 SID（+标题）→ 目标 SID。"""
    hist = [i for i in history if i in sid_table]
    if not hist or target not in sid_table:
        return None

    lines = []
    for idx, item in enumerate(hist, 1):
        sid = sid_table.sid(item)
        if with_titles:
            title = str((meta.get(str(item)) or {}).get("title") or "")[:80]
            lines.append(f"{idx}. {sid} {title}")
        else:
            lines.append(f"{idx}. {sid}")

    user_msg = (
        f"The user has interacted with the following {category_prompt} in "
        f"chronological order:\n" + "\n".join(lines) + "\n\n"
        "Predict the Semantic ID of the next item."
    )
    return {
        "task": "seqrec",
        "user_id": str(user_id),
        "history": [str(i) for i in hist],
        "target_item": str(target),
        "prompt": [
            {"role": "system", "content": SYSTEM_SID},
            {"role": "user", "content": user_msg},
        ],
        "answer": sid_table.sid(target),
    }


def sid_title2sid_example(
    *, item: str, sid_table: Any, meta: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """辅助任务：标题 → SID。让新增的 SID token 和物品语义对齐。"""
    title = str((meta.get(str(item)) or {}).get("title") or "").strip()
    if not title or item not in sid_table:
        return None
    return {
        "task": "title2sid",
        "user_id": "",
        "history": [],
        "target_item": str(item),
        "prompt": [
            {"role": "system", "content": SYSTEM_SID},
            {"role": "user", "content": f"What is the Semantic ID of the product titled: {title}?"},
        ],
        "answer": sid_table.sid(item),
    }


def sid_sid2title_example(
    *, item: str, sid_table: Any, meta: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """辅助任务：SID → 标题。反向对齐。"""
    title = str((meta.get(str(item)) or {}).get("title") or "").strip()
    if not title or item not in sid_table:
        return None
    return {
        "task": "sid2title",
        "user_id": "",
        "history": [],
        "target_item": str(item),
        "prompt": [
            {"role": "system", "content": SYSTEM_SID},
            {
                "role": "user",
                "content": f"What is the product title of Semantic ID {sid_table.sid(item)}?",
            },
        ],
        "answer": title,
    }


# --------------------------------------------------------- 路线 2：检索 query


def query_gen_example(
    *,
    user_id: str,
    history: Sequence[str],
    target: str,
    meta: dict[str, dict[str, Any]],
    max_history: int = 10,
) -> dict[str, Any] | None:
    """Rec-R1：历史 → 检索 query。

    SFT 的监督目标用目标物品标题构造一个冷启动 query（原文没有 SFT，
    这是我们为了统一基线加的）。RL 阶段只用 prompt，answer 用不到。
    """
    hist = [i for i in history if str(i) in meta][-max_history:]
    if not hist or str(target) not in meta:
        return None

    titles = _titles(hist, meta)
    user_msg = (
        "The user has purchased:\n"
        + "\n".join(f"- {t}" for t in titles)
        + "\n\nWrite a search query to retrieve the next item this user will buy."
    )
    target_title = str((meta.get(str(target)) or {}).get("title") or "")
    answer = (
        f"<think>The user's history suggests interest in this product category.</think>"
        f"<answer>{json.dumps({'query': target_title[:120]}, ensure_ascii=False)}</answer>"
    )
    return {
        "task": "query_gen",
        "user_id": str(user_id),
        "history": [str(i) for i in hist],
        "target_item": str(target),
        "prompt": [
            {"role": "system", "content": SYSTEM_QUERY},
            {"role": "user", "content": user_msg},
        ],
        "answer": answer,
    }


# ------------------------------------------------------- 路线 3：偏好推理文本

# 论文 Fig.3 的场景因子。原文是电影域（genre/director/actors/…），
# 这里换成 Amazon 商品域的对应维度。
SCENARIO_FACTORS = [
    "product category",
    "brand",
    "price tier",
    "use case",
    "quality expectation",
    "feature preference",
]


def preference_reasoning_example(
    *,
    user_id: str,
    history: Sequence[str],
    target: str,
    meta: dict[str, dict[str, Any]],
    max_history: int = 25,
) -> dict[str, Any] | None:
    """DPO4Rec：历史 → 结构化的用户偏好推理文本。

    prompt 模板对齐论文 Fig.3 的三段式（角色 + 用户信息/交互物品 + 场景因子）。
    """
    hist = [i for i in history if str(i) in meta][-max_history:]
    if not hist or str(target) not in meta:
        return None

    lines = []
    for item in hist:
        entry = meta.get(str(item)) or {}
        title = str(entry.get("title") or "")[:80]
        brand = str(entry.get("brand") or "")
        lines.append(f"- {title}" + (f" (brand: {brand})" if brand else ""))

    user_msg = (
        "Given a user whose purchase history over time is listed below:\n"
        + "\n".join(lines)
        + "\n\nAnalyze the user's preferences, considering these factors: "
        + ", ".join(SCENARIO_FACTORS)
        + ".\nProvide clear explanations based on details from the user's history "
        "and other pertinent factors."
    )
    return {
        "task": "preference_reasoning",
        "user_id": str(user_id),
        "history": [str(i) for i in hist],
        "target_item": str(target),
        "prompt": [
            {"role": "system", "content": SYSTEM_REASONING},
            {"role": "user", "content": user_msg},
        ],
        # 偏好推理没有 ground truth 文本。SFT 阶段用"按因子分条"的模板化文本
        # 做格式预热，保证 DPO 采样的 N 份候选差异来自内容而不是格式。
        "answer": _template_reasoning(lines, meta, hist),
    }


def _template_reasoning(
    lines: list[str], meta: dict[str, dict[str, Any]], history: Sequence[str]
) -> str:
    brands = [str((meta.get(str(i)) or {}).get("brand") or "") for i in history]
    brands = [b for b in brands if b]
    cats = [str((meta.get(str(i)) or {}).get("categories") or "") for i in history]
    cats = [c for c in cats if c]
    top_brand = max(set(brands), key=brands.count) if brands else "no dominant brand"
    top_cat = max(set(cats), key=cats.count) if cats else "mixed categories"
    return (
        f"1. Product category: the user consistently buys within {top_cat}.\n"
        f"2. Brand: {top_brand} appears most often, suggesting brand affinity.\n"
        f"3. Use case: the purchases cluster around a recurring practical need.\n"
        f"4. Feature preference: the user favors items matching the attributes "
        f"seen across {len(lines)} prior purchases."
    )


# ------------------------------------------------------------------ 构建入口


def build_examples(
    route: str,
    *,
    sequences: dict[str, list[dict[str, Any]]],
    meta: dict[str, dict[str, Any]],
    split: str,
    cfg: dict[str, Any],
    sid_table: Any = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """按路线 + split 构建样本。

    train：用户序列里除最后两条外的每个位置都可以当预测点（滑窗）；
    val/test：各取一条（leave-one-out）。
    """
    data = cfg["data"]
    history_max = int(data.get("history_max_length") or 20)
    category_prompt = str(data.get("category_prompt") or "items")
    out: list[dict[str, Any]] = []

    for user, events in sequences.items():
        items = [str(e["item_id"]) for e in events]
        splits = [str(e.get("split") or "train") for e in events]

        if split == "train":
            # 训练：所有 train 位置做滑窗
            positions = [
                i for i in range(1, len(items)) if splits[i] == "train"
            ]
        else:
            positions = [i for i in range(len(items)) if splits[i] == split]

        for pos in positions:
            history = items[max(0, pos - history_max) : pos]
            target = items[pos]
            if not history:
                continue
            example = _dispatch(
                route,
                user_id=user,
                history=history,
                target=target,
                meta=meta,
                sid_table=sid_table,
                category_prompt=category_prompt,
            )
            if example is not None:
                example["split"] = split
                if route == "dpo4rec":
                    n_pos = int(
                        ((cfg.get("decoder") or {}).get("reranker") or {}).get(
                            "n_positives"
                        )
                        or 4
                    )
                    # KAR ``rerank_item_from_hist``: next n interactions are relevant
                    example["positive_items"] = items[pos : pos + n_pos]
                out.append(example)
                if limit is not None and len(out) >= limit:
                    return out
    return out


def _dispatch(
    route: str,
    *,
    user_id: str,
    history: list[str],
    target: str,
    meta: dict[str, dict[str, Any]],
    sid_table: Any,
    category_prompt: str,
) -> dict[str, Any] | None:
    if route == "minionerec":
        if sid_table is None:
            raise ConfigurationError("minionerec 路线需要 sid_table")
        return sid_seqrec_example(
            user_id=user_id,
            history=history,
            target=target,
            sid_table=sid_table,
            meta=meta,
            category_prompt=category_prompt,
        )
    if route == "recr1":
        return query_gen_example(
            user_id=user_id, history=history, target=target, meta=meta
        )
    if route == "dpo4rec":
        return preference_reasoning_example(
            user_id=user_id, history=history, target=target, meta=meta
        )
    raise ConfigurationError(f"未知路线 '{route}'")


def build_auxiliary_examples(
    *, sid_table: Any, meta: dict[str, dict[str, Any]], tasks: Sequence[str]
) -> list[dict[str, Any]]:
    """MiniOneRec 的 title2sid / sid2title 辅助任务（只用于 SFT）。"""
    out: list[dict[str, Any]] = []
    for item in sid_table.items():
        if "title2sid" in tasks:
            ex = sid_title2sid_example(item=item, sid_table=sid_table, meta=meta)
            if ex:
                ex["split"] = "train"
                out.append(ex)
        if "sid2title" in tasks:
            ex = sid_sid2title_example(item=item, sid_table=sid_table, meta=meta)
            if ex:
                ex["split"] = "train"
                out.append(ex)
    return out
