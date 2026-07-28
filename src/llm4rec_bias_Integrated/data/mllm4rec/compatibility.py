# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
#
# Original behavior is preserved unless explicitly documented.

"""Official schema load / validate / optional lab conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from llm4rec_bias_Integrated.data.mllm4rec.schemas import (
    OFFICIAL_PREPROCESS_REQUIRED_KEYS,
    OFFICIAL_RANKER_REQUIRED_KEYS,
    OfficialDatasetDict,
)
from llm4rec_bias_Integrated.data.mllm4rec.serializer import load_pickle

CompatibilityMode = Literal["original", "robust"]


def load_official_compatible_dataset(path: str | Path) -> OfficialDatasetDict:
    """Load a MLLM4Rec-compatible ``dataset.pkl``."""
    return load_pickle(path)  # type: ignore[return-value]


def validate_official_schema(
    dataset: dict[str, Any],
    *,
    require_captions: bool = False,
) -> list[str]:
    """Return a list of schema problems (empty == OK).

    Does not mutate ``dataset``. Does not invent fields.
    """
    errors: list[str] = []
    required = (
        OFFICIAL_RANKER_REQUIRED_KEYS
        if require_captions
        else OFFICIAL_PREPROCESS_REQUIRED_KEYS
    )
    for key in required:
        if key not in dataset:
            errors.append(f"missing required key: {key}")

    for key in ("train", "val", "test", "meta", "umap", "smap"):
        if key in dataset and not isinstance(dataset[key], dict):
            errors.append(f"{key} must be a dict, got {type(dataset[key]).__name__}")

    if "meta_img_des" in dataset:
        if not isinstance(dataset["meta_img_des"], dict):
            errors.append("meta_img_des must be a dict")
        elif "meta" in dataset and isinstance(dataset["meta"], dict):
            if set(dataset["meta"].keys()) != set(dataset["meta_img_des"].keys()):
                errors.append("meta and meta_img_des key sets differ")

    # Padding reserved: densified ids start at 1.
    for map_name in ("umap", "smap"):
        mapping = dataset.get(map_name)
        if isinstance(mapping, dict) and mapping:
            vals = list(mapping.values())
            if min(vals) < 1:
                errors.append(f"{map_name} contains id < 1 (0 is reserved for padding)")

    return errors


def convert_to_llm4rec_bias_schema(dataset: dict[str, Any]) -> dict[str, Any]:
    """Produce a *separate* lab-oriented view; does not alter official fields.

    Official ``meta`` / ``meta_img_des`` semantics are left untouched in the
    returned copy's official keys. Lab-specific fields live under
    ``extended_metadata`` / top-level convenience keys only as additives.
    """
    out = dict(dataset)
    train = dataset.get("train") or {}
    sequences = {
        str(uid): [str(i) for i in items] for uid, items in train.items()
    }
    extended = dict(out.get("extended_metadata") or {})
    extended["lab_sequences_train"] = sequences
    extended["schema"] = "llm4rec_bias_Integrated_mllm4rec_bridge_v1"
    out["extended_metadata"] = extended
    return out


def simulate_official_retriever_load(dataset: dict[str, Any]) -> dict[str, Any]:
    """Fields required by official ``LRUDataloader`` (no meta / captions)."""
    train = dataset["train"]
    val = dataset["val"]
    test = dataset["test"]
    umap = dataset["umap"]
    smap = dataset["smap"]
    user_count = len(umap)
    item_count = len(smap)
    # Padding id 0 must not collide with densified ids.
    if item_count < 1 or min(smap.values()) < 1:
        raise ValueError("smap must densify from 1 (0 reserved for padding)")
    # One eval-style history: train+val for user 1
    u = 1
    seq = list(train[u]) + list(val[u])
    return {
        "user_count": user_count,
        "item_count": item_count,
        "sample_user": u,
        "sample_seq_len": len(seq),
        "sample_target": test[u][0],
    }


def simulate_official_ranker_prompt(
    dataset: dict[str, Any],
    *,
    max_title_chars: int = 32,
) -> dict[str, Any]:
    """Build a prompt fragment like official ``seq_to_token_ids`` formatting.

    Uses ``meta`` + ``meta_img_des`` as ``{title : caption}``. Does not require
    ``retrieved.pkl`` (that is produced by the Retriever training stage).
    """
    errors = validate_official_schema(dataset, require_captions=True)
    if errors:
        raise ValueError(f"ranker schema invalid: {errors}")
    meta = dataset["meta"]
    des = dataset["meta_img_des"]
    u = 1
    hist = list(dataset["train"][u])[-3:]
    label = dataset["test"][u][0]
    candidates = [label] + [i for i in hist if i != label][:3]
    while len(candidates) < 4:
        # pad with other meta keys
        for k in meta:
            if k not in candidates:
                candidates.append(k)
            if len(candidates) >= 4:
                break

    def trunc(s: str) -> str:
        s = str(s)
        return s if len(s) <= max_title_chars else s[:max_title_chars]

    seq_t = " \n ".join(
        f"({idx + 1}) {{{trunc(meta[i])} : {trunc(des[i])}}}"
        for idx, i in enumerate(hist)
    )
    can_t = " \n ".join(
        f"({chr(ord('A') + idx)}) {{{trunc(meta[i])} : {trunc(des[i])}}}"
        for idx, i in enumerate(candidates)
    )
    return {
        "user": u,
        "history_prompt": seq_t,
        "candidates_prompt": can_t,
        "label_letter": chr(ord("A") + candidates.index(label)),
        "history_ids": hist,
        "candidate_ids": candidates,
    }
