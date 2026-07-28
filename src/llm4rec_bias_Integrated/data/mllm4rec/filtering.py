# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
#
# Original behavior is preserved unless explicitly documented.

"""User/item frequency filtering (official iterative k-core style)."""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

logger = logging.getLogger("llm4rec_bias_Integrated.mllm4rec")

FilterMode = Literal["original", "robust"]


def filter_triplets(
    df: pd.DataFrame,
    *,
    min_uc: int,
    min_sc: int,
    min_rating: int = 0,
    mode: FilterMode = "original",
    iterative_kcore: bool | None = None,
) -> pd.DataFrame:
    """Filter interactions by user/item minimum counts.

    Official ``filter_triplets`` (original):
    - Does **not** apply ``min_rating`` as a rating threshold (path-name only).
    - Iteratively drops items with count < min_sc and users with count < min_uc
      until stable when ``min_sc > 1 or min_uc > 1``.

    Robust may optionally apply ``min_rating`` when ``mode == "robust"``.
    """
    if min_uc < 2:
        raise AssertionError("Need at least 2 ratings per user for validation and test")

    out = df.copy()
    apply_rating = mode == "robust" and min_rating > 0
    if apply_rating:
        out = out[out["rating"] >= min_rating]
        logger.info("robust: applied min_rating=%s filter", min_rating)
    elif min_rating > 0 and mode == "original":
        logger.debug(
            "original: min_rating=%s ignored as filter (official behavior)",
            min_rating,
        )

    # Official always uses the iterative loop when thresholds > 1.
    use_iterative = True if mode == "original" else (
        True if iterative_kcore is None else bool(iterative_kcore)
    )

    logger.info("Filtering triplets (mode=%s, iterative=%s)", mode, use_iterative)
    if not use_iterative:
        if min_sc > 1:
            item_sizes = out.groupby("sid").size()
            good_items = item_sizes.index[item_sizes >= min_sc]
            out = out[out["sid"].isin(good_items)]
        if min_uc > 1:
            user_sizes = out.groupby("uid").size()
            good_users = user_sizes.index[user_sizes >= min_uc]
            out = out[out["uid"].isin(good_users)]
        return out

    # --- official AbstractDataset.filter_triplets ---
    if min_sc > 1 or min_uc > 1:
        item_sizes = out.groupby("sid").size()
        good_items = item_sizes.index[item_sizes >= min_sc]
        user_sizes = out.groupby("uid").size()
        good_users = user_sizes.index[user_sizes >= min_uc]
        while len(good_items) < len(item_sizes) or len(good_users) < len(user_sizes):
            if min_sc > 1:
                item_sizes = out.groupby("sid").size()
                good_items = item_sizes.index[item_sizes >= min_sc]
                out = out[out["sid"].isin(good_items)]

            if min_uc > 1:
                user_sizes = out.groupby("uid").size()
                good_users = user_sizes.index[user_sizes >= min_uc]
                out = out[out["uid"].isin(good_users)]

            item_sizes = out.groupby("sid").size()
            good_items = item_sizes.index[item_sizes >= min_sc]
            user_sizes = out.groupby("uid").size()
            good_users = user_sizes.index[user_sizes >= min_uc]
    return out
