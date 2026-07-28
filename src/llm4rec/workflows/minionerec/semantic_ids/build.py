"""Build semantic IDs + SID jsonl datasets from MovieLens processed artifacts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from llm4rec.core.exceptions import ConfigurationError, MissingArtifactError
from llm4rec.core.reproducibility import write_json
from llm4rec.components.model._impl.base import require_cuda
from llm4rec.components.prompts.sid import SYSTEM_PROMPT, build_sid_messages
from llm4rec.workflows.minionerec.semantic_ids.residual_kmeans import (
    DEFAULT_ENCODER,
    break_collisions,
    embed_texts,
    residual_quantize,
)
from llm4rec.workflows.minionerec.semantic_ids.table import SidTable


def sid_dir(processed_dir: Path) -> Path:
    path = processed_dir / "sid"
    path.mkdir(parents=True, exist_ok=True)
    return path


def item_text_from_meta(meta: dict[str, Any]) -> str:
    title = str(meta.get("title") or "")
    genres = meta.get("genres") or []
    if isinstance(genres, str):
        g = genres
    else:
        g = ", ".join(str(x) for x in genres) or "unknown"
    return f"{title}. Genres: {g}."


def build_semantic_ids(
    *,
    processed_dir: Path,
    levels: int = 3,
    codebook_size: int = 64,
    seed: int = 0,
    encoder: str = DEFAULT_ENCODER,
    device: str | None = None,
) -> Path:
    """Embed item texts → residual k-means → write ``sid/semantic_ids.json``."""
    require_cuda()
    meta_path = processed_dir / "item_meta.json"
    if not meta_path.is_file():
        raise MissingArtifactError(f"missing {meta_path}; run prepare first")
    with meta_path.open(encoding="utf-8") as f:
        item_meta = json.load(f)
    ids = sorted(item_meta.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    texts = [item_text_from_meta(item_meta[i]) for i in ids]
    dev = device or "cuda"
    X = embed_texts(texts, encoder=encoder, device=dev)
    codes = residual_quantize(X, levels, codebook_size, seed)
    codes = break_collisions(codes)
    n_coll = int(codes[:, -1].max()) + 1
    uniq = len({tuple(c) for c in codes})
    if uniq != len(ids):
        raise ConfigurationError("SID collision breaking failed to uniquify IDs")

    out = sid_dir(processed_dir) / "semantic_ids.json"
    payload = {
        "levels": levels,
        "K": codebook_size,
        "collision_K": n_coll,
        "encoder": encoder,
        "items": {str(i): [int(x) for x in c] for i, c in zip(ids, codes)},
    }
    write_json(out, payload)
    return out


def _load_user_sequences(processed_dir: Path) -> dict[str, list[str]]:
    """Chronological item sequences from interactions.jsonl."""
    path = processed_dir / "interactions.jsonl"
    if not path.is_file():
        raise MissingArtifactError(f"missing {path}")
    by_user: dict[str, list[tuple[int, str]]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            uid = str(row["user_id"])
            by_user.setdefault(uid, []).append((int(row["timestamp"]), str(row["item_id"])))
    return {
        u: [i for _, i in sorted(events, key=lambda t: t[0])]
        for u, events in by_user.items()
    }


def _pop_quantiles(processed_dir: Path) -> dict[str, float]:
    path = processed_dir / "popularity.json"
    if not path.is_file():
        raise MissingArtifactError(f"missing {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if "quantiles" in data and isinstance(data["quantiles"], dict):
        return {str(k): float(v) for k, v in data["quantiles"].items()}
    out: dict[str, float] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[str(k)] = float(v.get("quantile", v.get("pop_quantile", 0.5)))
        else:
            out[str(k)] = float(v)
    return out


def _pop_counts(processed_dir: Path) -> dict[str, int]:
    path = processed_dir / "popularity.json"
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if "counts" in data and isinstance(data["counts"], dict):
        return {str(k): int(v) for k, v in data["counts"].items()}
    out: dict[str, int] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[str(k)] = int(v.get("count", 0))
        else:
            out[str(k)] = 0
    return out


def build_example(
    *,
    user_id: str,
    hist_items: list[str],
    target: str,
    titles: dict[str, str],
    table: SidTable,
    popq: dict[str, float],
    history_len: int,
    with_titles: bool,
) -> dict[str, Any] | None:
    hist = [i for i in hist_items if i in table.codes][-history_len:]
    if len(hist) < 2 or target not in table.codes:
        return None
    messages = build_sid_messages(
        history_item_ids=hist,
        titles=titles if with_titles else None,
        table=table,
        with_titles=with_titles,
    )
    return {
        "prompt": messages,
        "answer": table.sid(target),
        "target_item": target,
        "pop_quantile": float(popq.get(target, 0.5)),
        "hist_pop_mean": float(sum(popq.get(i, 0.5) for i in hist) / len(hist)),
        "user": user_id,
        "semantic_id": list(table.codes[target]),
    }


def build_sid_dataset(
    *,
    processed_dir: Path,
    sid_table_path: Path | None = None,
    history_max_length: int = 8,
    train_per_user: int = 4,
    seed: int = 0,
    with_titles: bool = True,
    train_limit: int | None = None,
    eval_limit: int | None = None,
) -> dict[str, Path]:
    """Write sid_{train,val,test}.jsonl + sid/item_meta.json."""
    out_dir = sid_dir(processed_dir)
    table_path = sid_table_path or (out_dir / "semantic_ids.json")
    if not table_path.is_file():
        raise MissingArtifactError(f"missing SID table {table_path}")
    table = SidTable(table_path)

    with (processed_dir / "item_meta.json").open(encoding="utf-8") as f:
        raw_meta = json.load(f)
    titles = {str(k): str(v.get("title") or "") for k, v in raw_meta.items()}
    popq = _pop_quantiles(processed_dir)
    counts = _pop_counts(processed_dir)

    sid_meta = {
        i: {
            "title": titles.get(i, ""),
            "pop_quantile": float(popq.get(i, 0.5)),
            "count": int(counts.get(i, 0)),
        }
        for i in table.codes
    }
    write_json(out_dir / "item_meta.json", sid_meta)

    seqs = _load_user_sequences(processed_dir)
    rng = random.Random(seed)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for user, s in seqs.items():
        if len(s) < 5:
            continue
        pos_choices = list(range(2, len(s) - 2))
        if pos_choices:
            for t in rng.sample(pos_choices, min(train_per_user, len(pos_choices))):
                ex = build_example(
                    user_id=user,
                    hist_items=s[:t],
                    target=s[t],
                    titles=titles,
                    table=table,
                    popq=popq,
                    history_len=history_max_length,
                    with_titles=with_titles,
                )
                if ex:
                    splits["train"].append(ex)
        for name, (h, tgt) in {
            "val": (s[:-2], s[-2]),
            "test": (s[:-1], s[-1]),
        }.items():
            ex = build_example(
                user_id=user,
                hist_items=h,
                target=tgt,
                titles=titles,
                table=table,
                popq=popq,
                history_len=history_max_length,
                with_titles=with_titles,
            )
            if ex:
                splits[name].append(ex)

    paths: dict[str, Path] = {}
    for name, rows in splits.items():
        rng.shuffle(rows)
        if name == "train" and train_limit is not None:
            rows = rows[: int(train_limit)]
        if name != "train" and eval_limit is not None:
            rows = rows[: int(eval_limit)]
        path = out_dir / f"sid_{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths[name] = path
    return paths


def load_sid_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= int(limit):
                break
    return rows
