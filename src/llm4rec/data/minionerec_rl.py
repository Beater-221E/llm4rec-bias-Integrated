"""MiniOneRec reproduction RL datasets — exact upstream prompts / targets.

Pinned upstream: ``AkaliKong/MiniOneRec @ 0c64b955`` (``rl.py`` → ``data.py``).

Active upstream mixture:
  - ``SidDataset`` (sample=-1)
  - ``RLTitle2SidDataset`` (sample=-1; title2sid + description2sid)
  - ``RLSeqTitle2SidDataset`` (sample=10000)

Each example carries ``prompt`` (raw string, no chat template) and
``reference_target_text`` (SID string + ``"\\n"``) for reference reward.
"""

from __future__ import annotations

import random
import re
from typing import Any, Sequence

from llm4rec.data.minionerec_prompts import (
    description2sid_user_input,
    format_minionerec_rl_prompt,
    seq_title2sid_user_input,
    sid_sft_user_input,
    title2sid_user_input,
)


def _sid_str(sid_table: Any, item: str) -> str:
    return str(sid_table.sid(item))


def _description_of(meta_entry: dict[str, Any], title: str) -> str:
    desc = meta_entry.get("description")
    if isinstance(desc, str) and desc.startswith("['") and desc.endswith("']"):
        try:
            parts = eval(desc)
            if parts:
                return str(parts[0])
        except Exception:
            pass
    if isinstance(desc, list):
        desc = next((d for d in desc if d and str(d).strip()), "")
    desc = str(desc or "").strip()
    return desc or title


# ------------------------------------------------------------------ SidDataset


def sid_rl_example(
    *, history_sids: Sequence[str], target_sid: str, user_id: str = ""
) -> dict[str, Any]:
    prompt = format_minionerec_rl_prompt(sid_sft_user_input(history_sids))
    return {
        "objective": "sid",
        "prompt": prompt,
        "reference_target_text": target_sid + "\n",
        "target_sid": target_sid,
        "user_id": str(user_id),
        "split": "train",
    }


# ------------------------------------------------------------ RLTitle2SidDataset


def rl_title2sid_examples(
    *, sid_table: Any, meta: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    title_to_sid: dict[str, str] = {}
    desc_to_sid: dict[str, str] = {}
    for item in sid_table.items():
        m = meta.get(str(item)) or {}
        title = str(m.get("title") or "").strip()
        if not title or item not in sid_table:
            continue
        sid = _sid_str(sid_table, item)
        title_to_sid[title] = sid
        desc = _description_of(m, title)
        if desc:
            desc_to_sid[desc] = sid
    out: list[dict[str, Any]] = []
    for title, sid in title_to_sid.items():
        out.append(
            {
                "objective": "title2sid",
                "prompt": format_minionerec_rl_prompt(title2sid_user_input(title)),
                "reference_target_text": sid + "\n",
                "target_sid": sid,
                "user_id": "",
                "split": "train",
            }
        )
    for desc, sid in desc_to_sid.items():
        out.append(
            {
                "objective": "description2sid",
                "prompt": format_minionerec_rl_prompt(description2sid_user_input(desc)),
                "reference_target_text": sid + "\n",
                "target_sid": sid,
                "user_id": "",
                "split": "train",
            }
        )
    return out


# ---------------------------------------------------------- RLSeqTitle2SidDataset


def rl_seq_title2sid_example(
    *, history_titles: Sequence[str], target_sid: str, user_id: str = ""
) -> dict[str, Any]:
    inter_titles = ", ".join(f'"{t}"' for t in history_titles)
    prompt = format_minionerec_rl_prompt(seq_title2sid_user_input(inter_titles))
    return {
        "objective": "seq_title2sid",
        "prompt": prompt,
        "reference_target_text": target_sid + "\n",
        "target_sid": target_sid,
        "user_id": str(user_id),
        "split": "train",
    }


# ------------------------------------------------------------------- builders


def build_minionerec_reproduction_rl_train(
    *,
    train_rows: Sequence[dict[str, Any]],
    sid_table: Any,
    meta: dict[str, dict[str, Any]],
    datasets: dict[str, dict[str, Any]] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Mix Sid + RLTitle2Sid + RLSeqTitle2Sid with upstream sampling.

    ``train_rows`` preserve every sliding window (no user dedup).
    """
    ds_cfg = {
        "sid_seq": {"sample": -1},
        "title_to_sid": {"sample": -1},
        "seq_title_to_sid": {"sample": 10000},
    }
    if datasets:
        for k, v in datasets.items():
            ds_cfg.setdefault(k, {}).update(v or {})

    rng = random.Random(int(seed))
    out: list[dict[str, Any]] = []

    # SidDataset
    n = int(ds_cfg["sid_seq"].get("sample") or -1)
    sid_rows: list[dict[str, Any]] = []
    for row in train_rows:
        hist = [i for i in (row.get("history") or []) if i in sid_table]
        target = row.get("target_item")
        if not hist or target not in sid_table:
            continue
        sid_rows.append(
            sid_rl_example(
                history_sids=[_sid_str(sid_table, i) for i in hist],
                target_sid=_sid_str(sid_table, target),
                user_id=str(row.get("user_id") or ""),
            )
        )
    if 0 < n < len(sid_rows):
        sid_rows = rng.sample(sid_rows, n)
    out.extend(sid_rows)

    # RLTitle2Sid
    n = int(ds_cfg["title_to_sid"].get("sample") or -1)
    t2s = rl_title2sid_examples(sid_table=sid_table, meta=meta)
    if 0 < n < len(t2s):
        t2s = rng.sample(t2s, n)
    out.extend(t2s)

    # RLSeqTitle2Sid
    n = int(ds_cfg["seq_title_to_sid"].get("sample") or 10000)
    seq_rows: list[dict[str, Any]] = []
    for row in train_rows:
        hist = [i for i in (row.get("history") or []) if i in sid_table]
        target = row.get("target_item")
        if not hist or target not in sid_table:
            continue
        titles = [str((meta.get(str(i)) or {}).get("title") or "").strip() for i in hist]
        seq_rows.append(
            rl_seq_title2sid_example(
                history_titles=titles,
                target_sid=_sid_str(sid_table, target),
                user_id=str(row.get("user_id") or ""),
            )
        )
    if 0 < n < len(seq_rows):
        seq_rows = rng.sample(seq_rows, n)
    out.extend(seq_rows)
    return out


def build_minionerec_reproduction_rl_eval(
    *,
    eval_rows: Sequence[dict[str, Any]],
    sid_table: Any,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in eval_rows:
        hist = [i for i in (row.get("history") or []) if i in sid_table]
        target = row.get("target_item")
        if not hist or target not in sid_table:
            continue
        out.append(
            sid_rl_example(
                history_sids=[_sid_str(sid_table, i) for i in hist],
                target_sid=_sid_str(sid_table, target),
                user_id=str(row.get("user_id") or ""),
            )
        )
    return out


def rl_dataset_counts(examples: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ex in examples:
        obj = str(ex.get("objective") or "unknown")
        counts[obj] = counts.get(obj, 0) + 1
    counts["total"] = len(examples)
    return counts
