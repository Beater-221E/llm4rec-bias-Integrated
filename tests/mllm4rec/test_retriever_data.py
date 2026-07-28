"""Retriever dataloader smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm4rec_bias_Integrated.mllm4rec.retriever.data import build_lru_loaders, load_official_dataset

PKL = Path("data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/dataset.pkl")


@pytest.mark.skipif(not PKL.is_file(), reason="dataset.pkl missing")
def test_lru_loaders_shapes():
    ds = load_official_dataset(PKL)
    train, val, test, n_u, n_i = build_lru_loaders(
        ds,
        max_len=50,
        sliding_window_size=1.0,
        train_batch_size=8,
        val_batch_size=8,
        test_batch_size=8,
    )
    assert n_u == 610 and n_i == 3650
    seqs, labels = next(iter(train))
    assert seqs.shape[1] == 50
    assert labels.shape == seqs.shape
    vseq, vans = next(iter(val))
    assert vans.shape[1] == 1
