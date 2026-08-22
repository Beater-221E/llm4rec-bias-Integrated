"""MiniOneRec SID distillation dataset + collator.

Mapped from ``dragonfly90/llm4rec-bias`` ``src/llm4rec/sid_distill.py``:

* ``DistillRows`` → ``MiniOneRecDistillDataset`` (prompt / history / target only)
* ``make_collate`` → ``MiniOneRecDistillCollator`` (Integrated chat + SidTable)

Teacher scoring and with-replacement sampling happen in the collator so the
dataset stays a thin row view. Soft items are **sampled from P***, not top-k.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from llm4rec.core.exceptions import ConfigurationError
from llm4rec.sid.table import sid_token

IGNORE_INDEX = -100


def _prompt_token_ids(tokenizer: Any, prompt: Sequence[dict[str, Any]]) -> list[int]:
    ids = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, tokenize=True
    )
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if isinstance(ids, str):
        ids = tokenizer(ids, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(t) for t in ids]


def item_sid_token_ids(tokenizer: Any, sid_table: Any, item: str) -> list[int]:
    """Full SID token ids using Integrated ``<a_*>`` / ``<b_*>`` / … format."""
    codes = sid_table.codes[str(item)]
    ids: list[int] = []
    for layer, code in enumerate(codes):
        token = sid_token(layer, int(code), sid_table.prefixes)
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is None or int(tid) < 0:
            raise ConfigurationError(f"SID token {token} 不在 tokenizer 词表里")
        ids.append(int(tid))
    return ids


class MiniOneRecDistillDataset(Dataset):
    """Returns prompt / history / target_item. No teacher calls here."""

    def __init__(self, examples: Sequence[dict[str, Any]], sid_table: Any) -> None:
        self.sid_table = sid_table
        self.examples: list[dict[str, Any]] = []
        for row in examples:
            history = list(row.get("history") or [])
            target = row.get("target_item")
            prompt = row.get("prompt")
            if not history or target is None or not prompt:
                continue
            if str(target) not in sid_table:
                continue
            self.examples.append(
                {
                    "prompt": prompt,
                    "history": [str(i) for i in history],
                    "target_item": str(target),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return dict(self.examples[int(idx)])


class MiniOneRecDistillCollator:
    """Tokenize ``prompt + SID(item)`` for M soft samples + 1 gold target.

    Labels cover SID completion tokens only. Also returns teacher level-1
    catalog-marginals (aligned with the sampling distribution).
    """

    def __init__(
        self,
        tokenizer: Any,
        sid_table: Any,
        teacher: Any,
        *,
        samples_per_prompt: int = 8,
        max_length: int = 1024,
        catalog_chunk_size: int = 256,
        generator: torch.Generator | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.sid_table = sid_table
        self.teacher = teacher
        self.samples_per_prompt = max(1, int(samples_per_prompt))
        self.max_length = int(max_length)
        self.catalog_chunk_size = max(1, int(catalog_chunk_size))
        self.generator = generator
        self.pad_token_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0)

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            raise ConfigurationError("distill collator 收到空 batch")
        histories = [row["history"] for row in rows]
        sampled, _logp, level1 = self.teacher.sample_items(
            histories,
            self.samples_per_prompt,
            generator=self.generator,
            chunk_size=self.catalog_chunk_size,
        )

        prompt_ids: list[torch.Tensor] = []
        completion_ids: list[torch.Tensor] = []
        soft_weight: list[float] = []
        gold_mask: list[bool] = []
        prompt_index: list[int] = []
        n_sid: list[int] = []
        weight = 1.0 / float(self.samples_per_prompt)

        for b, row in enumerate(rows):
            prompt = _prompt_token_ids(self.tokenizer, row["prompt"])
            candidates = list(sampled[b]) + [str(row["target_item"])]
            for j, item in enumerate(candidates):
                sid_ids = item_sid_token_ids(self.tokenizer, self.sid_table, item)
                full_prompt = list(prompt)
                labels_prompt = [IGNORE_INDEX] * len(full_prompt)
                input_ids = full_prompt + sid_ids
                labels = labels_prompt + list(sid_ids)
                if len(input_ids) > self.max_length:
                    overflow = len(input_ids) - self.max_length
                    input_ids = input_ids[overflow:]
                    labels = labels[overflow:]
                    kept_sid = sum(1 for x in labels if x != IGNORE_INDEX)
                    sid_ids = sid_ids[-kept_sid:] if kept_sid else []
                prompt_len = len(input_ids) - len(sid_ids)
                prompt_ids.append(torch.tensor(input_ids[:prompt_len], dtype=torch.long))
                completion_ids.append(torch.tensor(sid_ids, dtype=torch.long))
                is_gold = j == len(candidates) - 1
                soft_weight.append(0.0 if is_gold else weight)
                gold_mask.append(is_gold)
                prompt_index.append(b)
                n_sid.append(len(sid_ids))

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "soft_weight": torch.tensor(soft_weight, dtype=torch.float32),
            "gold_mask": torch.tensor(gold_mask, dtype=torch.bool),
            "prompt_index": torch.tensor(prompt_index, dtype=torch.long),
            "n_sid_tokens": torch.tensor(n_sid, dtype=torch.long),
            "level1_teacher": level1.detach().float().cpu(),
            "n_prompts": len(rows),
        }
