# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
#
# Original behavior is preserved unless explicitly documented.

"""User/item densification (internal ids start at 1; 0 = padding)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("llm4rec.workflows.mllm4rec._stack")


def densify_index(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[Any, int], dict[Any, int]]:
    """Map raw uid/sid to contiguous internal ids starting at 1.

    Official::
        umap = {u: i for i, u in enumerate(set(df['uid']), start=1)}
        smap = {s: i for i, s in enumerate(set(df['sid']), start=1)}
    """
    logger.info("Densifying index")
    umap = {u: i for i, u in enumerate(set(df["uid"]), start=1)}
    smap = {s: i for i, s in enumerate(set(df["sid"]), start=1)}
    out = df.copy()
    out["uid"] = out["uid"].map(umap)
    out["sid"] = out["sid"].map(smap)
    return out, umap, smap


def invert_map(mapping: dict[Any, int]) -> dict[int, Any]:
    """Invert raw→internal map; raises if not bijective."""
    inv: dict[int, Any] = {}
    for raw, internal in mapping.items():
        if internal in inv:
            raise ValueError(f"duplicate internal id {internal}")
        inv[internal] = raw
    if len(inv) != len(mapping):
        raise ValueError("map is not invertible")
    return inv


def assert_maps_invertible(umap: dict[Any, int], smap: dict[Any, int]) -> None:
    invert_map(umap)
    invert_map(smap)
    if umap and min(umap.values()) < 1:
        raise ValueError("umap must start at >= 1 (0 reserved for padding)")
    if smap and min(smap.values()) < 1:
        raise ValueError("smap must start at >= 1 (0 reserved for padding)")


def save_id_maps(
    directory: Path,
    *,
    umap: dict[Any, int],
    smap: dict[Any, int],
) -> tuple[Path, Path]:
    """Write user_id_map.json and item_id_map.json (raw → internal)."""
    directory.mkdir(parents=True, exist_ok=True)
    user_path = directory / "user_id_map.json"
    item_path = directory / "item_id_map.json"
    # JSON keys must be strings; preserve ints via str keys.
    user_path.write_text(
        json.dumps({str(k): int(v) for k, v in umap.items()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    item_path.write_text(
        json.dumps({str(k): int(v) for k, v in smap.items()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return user_path, item_path
