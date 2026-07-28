"""LOO split tests."""

from __future__ import annotations

import pandas as pd

from llm4rec_bias_Integrated.data.mllm4rec.splitting import split_df


def test_leave_one_out_and_timestamp_sid_order() -> None:
    df = pd.DataFrame(
        {
            "uid": [1, 1, 1, 1, 1],
            "sid": [5, 4, 3, 2, 1],
            "timestamp": [10, 10, 20, 30, 40],
            "rating": [5] * 5,
        }
    )
    # At timestamp 10, sid 4 then 5 after sort by timestamp,sid
    train, val, test = split_df(df, user_count=1, mode="original")
    assert test[1] == [1]
    assert val[1] == [2]
    assert train[1] == [4, 5, 3]


def test_no_split_leakage() -> None:
    df = pd.DataFrame(
        {
            "uid": [1] * 6 + [2] * 6,
            "sid": list(range(1, 7)) * 2,
            "timestamp": list(range(12)),
            "rating": [5] * 12,
        }
    )
    train, val, test = split_df(df, user_count=2, mode="original")
    for u in (1, 2):
        assert set(val[u]).isdisjoint(train[u])
        assert set(test[u]).isdisjoint(train[u])
        assert set(val[u]).isdisjoint(test[u])
        assert len(val[u]) == 1 and len(test[u]) == 1
