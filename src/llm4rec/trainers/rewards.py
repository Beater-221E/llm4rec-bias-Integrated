"""各路线的 reward 函数 —— 逐字对齐上游实现。

★ 三条路线的 reward 里都【没有】任何 bias 惩罚项。
  我们要观测的正是"纯准确率 reward 会不会放大 bias"，
  加了去偏项就把要测的现象抹掉了。做 mitigation 消融时另开配置。
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Sequence

from llm4rec.trainers.grpo import Rollout


# ============================================================ MiniOneRec


def make_minionerec_reward(sid_table: Any, cfg: dict[str, Any]):
    """对齐官方 ``rl.py`` 的 ``reward_type="ranking"``。

    官方是两个 reward 函数相加：

        rule_reward(i)      = 1.0 命中目标 SID / 0.0 未命中
        ndcg_rule_reward(i) = 组内第 i 条的位次奖励；官方写法是
                              w = [-1/log2(i+2)]，再整体除以 sum 取负，
                              等价于归一化后的 1/log2(i+2)，且只在命中时给。
    """
    reward_cfg = cfg.get("reward") or {}
    kind = str(reward_cfg.get("type") or "ranking")
    invalid_penalty = float(reward_cfg.get("invalid_penalty") or -1.0)

    def _positional_weights(n: int) -> list[float]:
        raw = [1.0 / math.log2(i + 2) for i in range(n)]
        total = sum(raw)
        return [w / total for w in raw] if total > 0 else [0.0] * n

    def reward_fn(rollout: Rollout) -> list[float]:
        target = str(rollout.example["target_item"])
        n = len(rollout.texts)
        weights = _positional_weights(n)
        out: list[float] = []
        for i, text in enumerate(rollout.texts):
            item = sid_table.parse(text)
            if item is None:
                out.append(invalid_penalty)
                continue
            hit = 1.0 if item == target else 0.0
            if kind == "rule":
                out.append(hit)
            elif kind == "ranking_only":
                out.append(weights[i] if hit else 0.0)
            else:  # ranking = rule + ndcg_rule
                out.append(hit + (weights[i] if hit else 0.0))
        return out

    return reward_fn


# ================================================================ Rec-R1

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_answer(text: str) -> str | None:
    """取最后一个 <answer>…</answer>（对齐官方 ``extract_solution``）。"""
    matches = _ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else None


def validate_structure(text: str) -> bool:
    """官方 ``validate_response_structure``：think/answer 标签各恰好一次。"""
    for tag in ("<think>", "</think>", "<answer>", "</answer>"):
        if text.count(tag) != 1:
            return False
    return _THINK_RE.search(text) is not None


def parse_query(answer: str | None, answer_key: str = "query") -> str | None:
    """官方 ``check_json_format``：answer 必须是含 query 键的合法 JSON。"""
    if not answer:
        return None
    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or answer_key not in data:
        return None
    query = str(data[answer_key]).strip()
    return query or None


def ndcg_at_k(retrieved: Sequence[str], target: str, k: int) -> float:
    """官方 ``ndcg_at_k``：只有一个相关物品，ideal DCG = 1。"""
    for rank, item in enumerate(retrieved[:k]):
        if str(item) == str(target):
            return float(1.0 / math.log2(rank + 2))
    return 0.0


def make_recr1_reward(retriever: Any, cfg: dict[str, Any], *, top_k: int | None = None):
    """对齐官方 ``verl/utils/reward_score/amazon_c4.py`` 的 ``compute_score``。

        format_score = +0.1 格式对 / -2 格式错
        answer_score = NDCG@top_k（检索结果 vs 目标 ASIN）；检索异常 = -2
        total = format + answer   若 answer > 0
              = 0                 若 answer <= 0 但格式对
              = format_score      若格式错（即 -2）
    """
    decoder_cfg = cfg.get("decoder") or {}
    reward_cfg = (cfg.get("train") or {}).get("rl", {}).get("reward") or {}
    answer_key = str(decoder_cfg.get("answer_key") or "query")
    retr_cfg = decoder_cfg.get("retriever") or {}
    k = int(top_k or retr_cfg.get("train_top_k") or 1000)

    format_reward = float(reward_cfg.get("format_reward") or 0.1)
    format_penalty = float(reward_cfg.get("format_penalty") or -2.0)
    error_score = float(reward_cfg.get("retrieval_error_score") or -2.0)

    def reward_fn(rollout: Rollout) -> list[float]:
        target = str(rollout.example["target_item"])
        out: list[float] = []
        for text in rollout.texts:
            answer = extract_answer(text)
            query = parse_query(answer, answer_key)
            format_ok = validate_structure(text) and query is not None
            format_score = format_reward if format_ok else format_penalty

            answer_score = 0.0
            if format_ok:
                try:
                    retrieved = retriever.search(query, top_k=k)
                    answer_score = ndcg_at_k(retrieved, target, k)
                except Exception:  # noqa: BLE001 — 对齐官方的 try/except
                    answer_score = error_score

            if answer_score > 0:
                out.append(format_score + answer_score)
            elif format_score > 0:
                out.append(0.0)
            else:
                out.append(format_score)
        return out

    return reward_fn


# ============================================================== DPO4Rec


def make_dpo4rec_scorer(reranker: Any, cfg: dict[str, Any]):
    """DPO4Rec 的"reward model" = 装了推理知识的 reranker。

    论文 §IV-B：把 LLM 生成的推理文本编码成向量，和 ID 表征拼接后喂给
    reranker，用它输出列表的 NDCG 给这份推理文本打分。分最高的当 chosen，
    最低的当 rejected。
    """
    reward_cfg = (cfg.get("train") or {}).get("dpo", {}).get("reward") or {}
    top_k = int(reward_cfg.get("top_k") or 5)

    def score_fn(example: dict[str, Any], reasoning_texts: Sequence[str]) -> list[float]:
        return [
            reranker.score_reasoning(example, text, top_k=top_k)
            for text in reasoning_texts
        ]

    return score_fn
