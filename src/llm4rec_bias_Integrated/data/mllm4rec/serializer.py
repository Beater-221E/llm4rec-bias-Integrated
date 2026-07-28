# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
#
# Original behavior is preserved unless explicitly documented.

"""Pickle / sidecar serialization for MLLM4Rec dataset objects."""

from __future__ import annotations

import json
import logging
import pickle
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger("llm4rec_bias_Integrated.mllm4rec")


def load_pickle(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(
    dataset: dict[str, Any],
    path: str | Path,
    *,
    atomic_write: bool = True,
    create_backup: bool = True,
) -> Path:
    """Write ``dataset.pkl``; optionally backup existing and write via ``.tmp``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if create_backup and path.is_file():
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)
        logger.info("Backed up existing pickle to %s", bak)

    if atomic_write:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(dataset, f)
        tmp.replace(path)
    else:
        with path.open("wb") as f:
            pickle.dump(dataset, f)
    logger.info("Wrote %s", path)
    return path


def save_id_map_sidecars(
    directory: Path,
    *,
    umap: dict[Any, int],
    smap: dict[Any, int],
) -> None:
    from llm4rec_bias_Integrated.data.mllm4rec.id_mapping import save_id_maps

    save_id_maps(directory, umap=umap, smap=smap)


def save_statistics_json(directory: Path, stats: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "dataset_statistics.json"
    path.write_text(json.dumps(stats, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def try_save_parquet_sidecars(
    directory: Path,
    dataset: dict[str, Any],
) -> list[Path]:
    """Best-effort parquet exports; failure is logged, never raises for missing deps."""
    written: list[Path] = []
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas unavailable; skip parquet sidecars")
        return written

    directory.mkdir(parents=True, exist_ok=True)
    meta = dataset.get("meta") or {}
    items_df = pd.DataFrame(
        [{"internal_item_id": k, "title": v} for k, v in meta.items()]
    )
    try:
        items_path = directory / "items.parquet"
        items_df.to_parquet(items_path, index=False)
        written.append(items_path)

        rows = []
        for split_name in ("train", "val", "test"):
            split = dataset.get(split_name) or {}
            for uid, items in split.items():
                for pos, iid in enumerate(items):
                    rows.append(
                        {
                            "split": split_name,
                            "user_id": uid,
                            "item_id": iid,
                            "position": pos,
                        }
                    )
        seq_path = directory / "sequences.parquet"
        pd.DataFrame(rows).to_parquet(seq_path, index=False)
        written.append(seq_path)
    except (ImportError, ValueError, OSError) as exc:
        logger.warning("Skip parquet sidecars: %s", exc)
    return written
