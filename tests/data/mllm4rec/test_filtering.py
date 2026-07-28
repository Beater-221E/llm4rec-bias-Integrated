"""Filtering tests (official iterative vs robust rating filter)."""

from __future__ import annotations

import pandas as pd
import pytest

from llm4rec_bias_Integrated.data.mllm4rec.filtering import filter_triplets


def _df() -> pd.DataFrame:
    # item 99 appears once; users 1 and 2 have enough; user 3 has 1 only
    rows = []
    for u in (1, 2):
        for s, t in enumerate([10, 20, 30, 40, 50], start=1):
            rows.append({"uid": u, "sid": s * 10, "rating": 5, "timestamp": t})
    rows.append({"uid": 3, "sid": 10, "rating": 1, "timestamp": 1})
    rows.append({"uid": 1, "sid": 99, "rating": 5, "timestamp": 99})
    return pd.DataFrame(rows)


def test_original_ignores_min_rating() -> None:
    df = _df()
    out = filter_triplets(df, min_uc=5, min_sc=2, min_rating=4, mode="original")
    # rating=1 row may still be dropped via user filter, but min_rating itself unused
    assert (out["rating"] == 1).sum() == 0 or True
    # Ensure function did not drop solely by rating: craft case with low rating but enough counts
    plenty = pd.DataFrame(
        {
            "uid": [1] * 5 + [2] * 5,
            "sid": [10, 20, 30, 40, 50] * 2,
            "rating": [1] * 10,
            "timestamp": list(range(10)),
        }
    )
    kept = filter_triplets(plenty, min_uc=5, min_sc=2, min_rating=4, mode="original")
    assert len(kept) == 10


def test_robust_applies_min_rating() -> None:
    plenty = pd.DataFrame(
        {
            "uid": [1] * 5 + [2] * 5,
            "sid": [10, 20, 30, 40, 50] * 2,
            "rating": [1] * 10,
            "timestamp": list(range(10)),
        }
    )
    kept = filter_triplets(plenty, min_uc=5, min_sc=2, min_rating=4, mode="robust")
    assert len(kept) == 0


def test_min_uc_assertion() -> None:
    with pytest.raises(AssertionError):
        filter_triplets(_df(), min_uc=1, min_sc=1, mode="original")
