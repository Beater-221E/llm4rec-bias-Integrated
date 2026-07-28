"""MovieLens-100k (ml-latest-small) parser tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from llm4rec_bias_Integrated.data.mllm4rec.config import MLLM4RecDataConfig
from llm4rec_bias_Integrated.data.mllm4rec.movielens_100k import (
    ML100KClassicDataset,
    ML100KDataset,
    clean_movielens_title_official,
)


def test_clean_title_official_basic() -> None:
    assert clean_movielens_title_official("Toy Story (1995)") == "Toy Story (1995)"


def test_clean_title_official_article_flip() -> None:
    # Official flips trailing ", The" style articles within last 5 chars of lower title.
    out = clean_movielens_title_official("Usual Suspects, The (1995)")
    assert out.endswith("(1995)")
    assert out.lower().startswith("the ")


def test_ml_latest_small_ratings_and_meta(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [10, 20, 30, 10, 20, 40],
            "rating": [4.0, 5.0, 3.0, 4.0, 4.0, 5.0],
            "timestamp": [1, 2, 3, 1, 2, 3],
        }
    ).to_csv(raw / "ratings.csv", index=False)
    pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40],
            "title": [
                "Toy Story (1995)",
                "Jumanji (1995)",
                "Grumpier Old Men (1995)",
                "Waiting to Exhale (1995)",
            ],
            "genres": ["Animation", "Adventure", "Comedy", "Comedy"],
        }
    ).to_csv(raw / "movies.csv", index=False)

    cfg = MLLM4RecDataConfig(
        dataset_code="ml-100k",
        raw_dir=raw,
        output_root=tmp_path / "preprocessed",
        min_uc=2,
        min_sc=1,
        save_parquet=False,
    )
    ds = ML100KDataset(cfg)
    ratings = ds.load_ratings_df()
    assert list(ratings.columns) == ["uid", "sid", "rating", "timestamp"]
    meta = ds.load_meta_dict()
    assert 10 in meta
    assert "Toy Story" in meta[10]


def test_classic_u_item_encoding(tmp_path: Path) -> None:
    raw = tmp_path / "ml-100k"
    raw.mkdir()
    # Latin-1 byte for é
    title = "Les Misérables (1995)".encode("latin-1")
    line = b"1|" + title + b"|01-Jan-1995||http://example.com|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0\n"
    (raw / "u.item").write_bytes(line)
    (raw / "u.data").write_text("1\t1\t5\t100\n", encoding="utf-8")
    cfg = MLLM4RecDataConfig(
        dataset_code="ml-100k-classic",
        raw_dir=raw,
        output_root=tmp_path / "out",
        min_uc=2,
        min_sc=1,
    )
    ds = ML100KClassicDataset(cfg)
    meta = ds.load_meta_dict()
    assert 1 in meta
    assert "Mis" in meta[1]
