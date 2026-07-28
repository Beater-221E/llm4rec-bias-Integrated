"""ID densification tests."""

from __future__ import annotations

import pandas as pd
import pytest

from llm4rec_bias_Integrated.data.mllm4rec.id_mapping import (
    assert_maps_invertible,
    densify_index,
    invert_map,
)


def test_densify_starts_at_one_and_invertible() -> None:
    df = pd.DataFrame(
        {
            "uid": [100, 100, 200],
            "sid": [9, 7, 9],
            "rating": [5, 4, 5],
            "timestamp": [1, 2, 3],
        }
    )
    out, umap, smap = densify_index(df)
    assert min(umap.values()) == 1
    assert min(smap.values()) == 1
    assert 0 not in umap.values()
    assert 0 not in smap.values()
    assert_maps_invertible(umap, smap)
    inv_u = invert_map(umap)
    assert inv_u[umap[100]] == 100
    assert set(out["uid"]) <= set(umap.values())


def test_padding_not_assigned() -> None:
    df = pd.DataFrame({"uid": [1], "sid": [1], "rating": [5], "timestamp": [1]})
    _, umap, smap = densify_index(df)
    assert set(umap.values()) == {1}
    assert set(smap.values()) == {1}
