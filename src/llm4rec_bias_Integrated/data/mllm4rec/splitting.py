# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
#
# Original behavior is preserved unless explicitly documented.

"""Leave-one-out sequence split (official split_df)."""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

logger = logging.getLogger("llm4rec_bias_Integrated.mllm4rec")

SplitMode = Literal["original", "robust"]


def split_df(
    df: pd.DataFrame,
    user_count: int,
    *,
    mode: SplitMode = "original",
) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, list[int]]]:
    """Per-user leave-one-out split.

    Official original:
    - sort by ``['timestamp', 'sid']``
    - train = items[:-2], val = items[-2:-1], test = items[-1:]
    - users keyed ``1 .. user_count``

    Robust may use ``['timestamp', 'source_row_index']`` when that column exists.
    """
    logger.info("Splitting (mode=%s)", mode)
    if mode == "robust" and "source_row_index" in df.columns:
        sort_cols = ["timestamp", "source_row_index"]
    else:
        sort_cols = ["timestamp", "sid"]

    # Equivalent to official groupby(...).progress_apply(sort then sid list),
    # without depending on tqdm.pandas().
    user2items: dict[int, list[int]] = {}
    for uid, group in df.groupby("uid"):
        ordered = group.sort_values(by=sort_cols, kind="mergesort")
        user2items[int(uid)] = list(ordered["sid"])

    train: dict[int, list[int]] = {}
    val: dict[int, list[int]] = {}
    test: dict[int, list[int]] = {}
    for i in range(user_count):
        user = i + 1
        items = user2items[user]
        train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
    return train, val, test
