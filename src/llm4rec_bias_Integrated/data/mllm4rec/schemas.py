# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
#
# Original behavior is preserved unless explicitly documented.

"""Official-compatible dataset.pkl field contract."""

from __future__ import annotations

from typing import Any, TypedDict


class OfficialDatasetDict(TypedDict, total=False):
    """Fields present in MLLM4Rec ``dataset.pkl``.

    After preprocess (ml-100k): train, val, test, meta, umap, smap.
    After BLIP2: also meta_img_des.
    Amazon datasets may also include meta_img_url (out of MovieLens scope).
    """

    train: dict[int, list[int]]
    val: dict[int, list[int]]
    test: dict[int, list[int]]
    meta: dict[int, str]
    umap: dict[Any, int]
    smap: dict[Any, int]
    meta_img_des: dict[int, str]
    meta_img_url: dict[int, str]
    extended_metadata: dict[str, Any]


OFFICIAL_PREPROCESS_REQUIRED_KEYS = (
    "train",
    "val",
    "test",
    "meta",
    "umap",
    "smap",
)

OFFICIAL_RANKER_REQUIRED_KEYS = OFFICIAL_PREPROCESS_REQUIRED_KEYS + ("meta_img_des",)
