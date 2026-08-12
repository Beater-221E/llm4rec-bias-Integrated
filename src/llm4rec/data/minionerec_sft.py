"""MiniOneRec reproduction SFT — exact upstream dataset/objective/tokenization.

Pinned upstream: ``AkaliKong/MiniOneRec @ 0c64b955`` (``sft.py`` → ``data.py``).

Differences vs integrated mode:
- NO chat template; raw tokenizer encoding with explicit BOS/EOS (upstream ``Tokenizer``).
- Left padding.
- Training rows are NOT collapsed by user (sliding windows preserved).
- ``SidItemFeat`` uses upstream dict-based sid2title/title2sid cardinality.
- Validation = ``SidSFT`` only.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from llm4rec.data.minionerec_prompts import (
    ALPACA_FUSION_INSTRUCTION,
    ALPACA_ITEMFEAT_INSTRUCTION,
    ALPACA_SFT_INSTRUCTION,
    format_minionerec_alpaca_prompt,
    fusion_prompt,
    sid2title_prompt,
    sid_sft_prompt,
    title2sid_prompt,
)

IGNORE_INDEX = -100

__all__ = [
    "ALPACA_FUSION_INSTRUCTION",
    "ALPACA_ITEMFEAT_INSTRUCTION",
    "ALPACA_SFT_INSTRUCTION",
    "IGNORE_INDEX",
    "MiniOneRecReferenceSFTDataset",
    "build_sft_rows",
    "encode_reference",
    "format_minionerec_alpaca_prompt",
    "fusion_prompt",
    "sid2title_prompt",
    "sid_sft_prompt",
    "sft_dataset_counts",
    "title2sid_prompt",
]


def encode_reference(
    tokenizer: Any,
    text: str,
    *,
    bos: bool = False,
    eos: bool = False,
) -> list[int]:
    """Upstream ``Tokenizer.encode``: raw text, strip duplicate BOS/EOS, explicit add."""
    ids = tokenizer.encode(text) if hasattr(tokenizer, "encode") else tokenizer(text)["input_ids"]
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    ids = list(ids)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if bos_id is not None:
        while ids and ids[0] == bos_id:
            ids = ids[1:]
    if eos_id is not None:
        while ids and ids[-1] == eos_id:
            ids = ids[:-1]
    if bos and bos_id is not None:
        ids = [int(bos_id)] + ids
    if eos and eos_id is not None:
        ids = ids + [int(eos_id)]
    return ids


def _sid_str(sid_table: Any, item: str) -> str:
    return str(sid_table.sid(item))


def build_sft_rows(
    *,
    train_rows: Sequence[dict[str, Any]],
    sid_table: Any,
    meta: dict[str, dict[str, Any]],
    objectives: Sequence[str],
    user_id_key: str = "user_id",
) -> list[dict[str, Any]]:
    """Build SidSFT + FusionSeqRec + SidItemFeat examples.

    ``train_rows`` may contain multiple windows per user; rows are never collapsed.
    Each row: ``{"history": [...], "target_item": ..., user_id?}``.
    SidItemFeat uses upstream dict mappings (duplicate title/sid collapse).
    """
    objs = list(objectives or ["sid_sft", "sid_item_feat", "fusion_seqrec"])
    out: list[dict[str, Any]] = []

    if "sid_sft" in objs or "fusion_seqrec" in objs:
        for row in train_rows:
            hist = [i for i in (row.get("history") or []) if i in sid_table]
            target = row.get("target_item")
            if not hist or target not in sid_table:
                continue
            sids = [_sid_str(sid_table, i) for i in hist]
            target_sid = _sid_str(sid_table, target)
            if "sid_sft" in objs:
                out.append(
                    {
                        "objective": "sid_sft",
                        "prompt_text": sid_sft_prompt(sids),
                        "answer_text": target_sid + "\n",
                        "user_id": str(row.get(user_id_key) or ""),
                        "target_item": str(target),
                        "target_sid": target_sid,
                        "split": "train",
                    }
                )
            if "fusion_seqrec" in objs:
                title = str((meta.get(str(target)) or {}).get("title") or "").strip() or target_sid
                out.append(
                    {
                        "objective": "fusion_seqrec",
                        "prompt_text": fusion_prompt(sids),
                        "answer_text": title + "\n",
                        "user_id": str(row.get(user_id_key) or ""),
                        "target_item": str(target),
                        "target_sid": target_sid,
                        "split": "train",
                    }
                )

    if "sid_item_feat" in objs:
        # Upstream SidItemFeatDataset: dict(sid->title), dict(title->sid)
        sid_to_title_ref: dict[str, str] = {}
        title_to_sid_ref: dict[str, str] = {}
        for item in sid_table.items():
            title = str((meta.get(str(item)) or {}).get("title") or "").strip()
            if not title or item not in sid_table:
                continue
            sid = _sid_str(sid_table, item)
            sid_to_title_ref[sid] = title
            title_to_sid_ref[title] = sid
        for sid, title in sid_to_title_ref.items():
            out.append(
                {
                    "objective": "sid_item_feat",
                    "prompt_text": sid2title_prompt(sid),
                    "answer_text": title + "\n",
                    "user_id": "",
                    "target_item": "",
                    "target_sid": sid,
                    "split": "train",
                }
            )
        for title, sid in title_to_sid_ref.items():
            out.append(
                {
                    "objective": "sid_item_feat",
                    "prompt_text": title2sid_prompt(title),
                    "answer_text": sid + "\n",
                    "user_id": "",
                    "target_item": "",
                    "target_sid": sid,
                    "split": "train",
                }
            )
    return out


def sft_dataset_counts(examples: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ex in examples:
        obj = str(ex.get("objective") or "unknown")
        counts[obj] = counts.get(obj, 0) + 1
    counts["total"] = len(examples)
    return counts


# --------------------------------------------------------------------- Dataset

# Backward-compat aliases for older example-level APIs.
# The reproduction path uses :func:`build_sft_rows`; these single-example helpers
# remain for tests / ad-hoc use.


def sid_sft_example(
    *,
    user_id: str,
    history: Sequence[str],
    target: str,
    sid_table: Any,
) -> dict[str, Any] | None:
    hist = [i for i in history if i in sid_table]
    if not hist or target not in sid_table:
        return None
    sids = [str(sid_table.sid(i)) for i in hist]
    return {
        "task": "sid_sft",
        "objective": "sid_sft",
        "user_id": str(user_id),
        "history": [str(i) for i in hist],
        "target_item": str(target),
        "prompt": [{"role": "user", "content": sid_sft_prompt(sids)}],
        "answer": str(sid_table.sid(target)) + "\n",
        "split": "train",
    }


def fusion_seqrec_example(
    *,
    user_id: str,
    history: Sequence[str],
    target: str,
    sid_table: Any,
    meta: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    hist = [i for i in history if i in sid_table]
    if not hist or target not in sid_table:
        return None
    sids = [str(sid_table.sid(i)) for i in hist]
    title = str((meta.get(str(target)) or {}).get("title") or "").strip() or str(sid_table.sid(target))
    return {
        "task": "fusion_seqrec",
        "objective": "fusion_seqrec",
        "user_id": str(user_id),
        "history": [str(i) for i in hist],
        "target_item": str(target),
        "prompt": [{"role": "user", "content": fusion_prompt(sids)}],
        "answer": title + "\n",
        "split": "train",
    }


def sid_item_feat_examples(
    *,
    item: str,
    sid_table: Any,
    meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    title = str((meta.get(str(item)) or {}).get("title") or "").strip()
    if not title or item not in sid_table:
        return []
    sid = str(sid_table.sid(item))
    return [
        {
            "task": "title2sid",
            "objective": "sid_item_feat",
            "user_id": "",
            "history": [],
            "target_item": str(item),
            "prompt": [{"role": "user", "content": title2sid_prompt(title)}],
            "answer": sid + "\n",
            "split": "train",
        },
        {
            "task": "sid2title",
            "objective": "sid_item_feat",
            "user_id": "",
            "history": [],
            "target_item": str(item),
            "prompt": [{"role": "user", "content": sid2title_prompt(sid)}],
            "answer": title + "\n",
            "split": "train",
        },
    ]


class MiniOneRecReferenceSFTDataset:
    """Raw-tokenized SFT dataset (no chat template). Left truncation tail."""

    def __init__(
        self,
        examples: Sequence[dict[str, Any]],
        tokenizer: Any,
        max_len: int = 512,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        prompt_ids = encode_reference(self.tokenizer, ex["prompt_text"], bos=True, eos=False)
        ans_ids = encode_reference(self.tokenizer, ex["answer_text"], bos=False, eos=True)
        input_ids = prompt_ids + ans_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + ans_ids
        attention_mask = [1] * len(input_ids)
        if len(input_ids) > self.max_len:  # keep tail like upstream [-max_len:]
            input_ids = input_ids[-self.max_len :]
            labels = labels[-self.max_len :]
            attention_mask = attention_mask[-self.max_len :]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
