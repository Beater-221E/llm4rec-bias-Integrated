"""Simulate official Retriever / Ranker field access on a real or toy pickle."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm4rec_bias_Integrated.data.mllm4rec.compatibility import (
    simulate_official_ranker_prompt,
    simulate_official_retriever_load,
    validate_official_schema,
)
from llm4rec_bias_Integrated.data.mllm4rec.serializer import load_pickle

REAL_PKL = Path(
    "data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/dataset.pkl"
)


@pytest.fixture
def toy_dataset():
    return {
        "train": {1: [1, 2, 3]},
        "val": {1: [4]},
        "test": {1: [5]},
        "meta": {i: f"Title {i} (1995)" for i in range(1, 6)},
        "meta_img_des": {i: f"caption {i}" for i in range(1, 6)},
        "umap": {10: 1},
        "smap": {100 + i: i for i in range(1, 6)},
    }


def test_retriever_and_ranker_on_toy(toy_dataset):
    assert validate_official_schema(toy_dataset, require_captions=True) == []
    r = simulate_official_retriever_load(toy_dataset)
    assert r["item_count"] == 5
    assert r["sample_target"] == 5
    p = simulate_official_ranker_prompt(toy_dataset)
    assert ":" in p["history_prompt"]
    assert p["label_letter"] == "A"


@pytest.mark.skipif(not REAL_PKL.is_file(), reason="full dataset.pkl not present")
def test_real_ml100k_official_loaders():
    ds = load_pickle(REAL_PKL)
    assert validate_official_schema(ds, require_captions=True) == []
    r = simulate_official_retriever_load(ds)
    assert r["user_count"] == 610
    assert r["item_count"] == 3650
    p = simulate_official_ranker_prompt(ds)
    assert len(p["history_prompt"]) > 0
    assert set(ds["meta"]) == set(ds["meta_img_des"])
