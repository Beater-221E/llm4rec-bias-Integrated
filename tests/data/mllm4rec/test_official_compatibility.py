"""End-to-end preprocess on synthetic ml-latest-small files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm4rec_bias_Integrated.data.mllm4rec.compatibility import validate_official_schema
from llm4rec_bias_Integrated.data.mllm4rec.config import MLLM4RecDataConfig
from llm4rec_bias_Integrated.data.mllm4rec.dataset_factory import dataset_factory
from llm4rec_bias_Integrated.data.mllm4rec.serializer import load_pickle


def _write_synth(raw: Path, *, n_users: int = 4, n_items: int = 6, length: int = 6) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    ratings = []
    ts = 1
    for u in range(1, n_users + 1):
        for t in range(length):
            sid = (t % n_items) + 1
            ratings.append(
                {"userId": u, "movieId": sid * 10, "rating": 4.0, "timestamp": ts}
            )
            ts += 1
    pd.DataFrame(ratings).to_csv(raw / "ratings.csv", index=False)
    movies = []
    for i in range(1, n_items + 1):
        movies.append(
            {
                "movieId": i * 10,
                "title": f"Movie {i} (1995)",
                "genres": "Comedy",
            }
        )
    pd.DataFrame(movies).to_csv(raw / "movies.csv", index=False)
    (raw / "README").write_text("synth\n", encoding="utf-8")


def test_preprocess_official_schema(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_synth(raw)
    cfg = MLLM4RecDataConfig(
        dataset_code="ml-100k",
        raw_dir=raw,
        output_root=tmp_path / "preprocessed",
        min_rating=0,
        min_uc=5,
        min_sc=2,
        save_parquet=False,
        compatibility_mode="original",
        filtering_mode="original",
    )
    ds = dataset_factory(cfg)
    path = ds.preprocess(overwrite=True)
    assert path.is_file()
    dataset = load_pickle(path)
    assert validate_official_schema(dataset) == []
    assert "meta_img_des" not in dataset
    assert min(dataset["smap"].values()) >= 1
    assert set(dataset["meta"].keys()) == set(dataset["smap"].values())
    # Every user has val/test length 1
    for u in dataset["train"]:
        assert len(dataset["val"][u]) == 1
        assert len(dataset["test"][u]) == 1
        assert len(dataset["train"][u]) >= 1
    assert (path.parent / "user_id_map.json").is_file()
    assert (path.parent / "item_id_map.json").is_file()
    assert (path.parent / "dataset_statistics.json").is_file()
