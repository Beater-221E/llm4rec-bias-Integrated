"""MovieLens-1M parser tests."""

from __future__ import annotations

from pathlib import Path

from llm4rec_bias_Integrated.data.mllm4rec.config import MLLM4RecDataConfig
from llm4rec_bias_Integrated.data.mllm4rec.movielens_1m import ML1MDataset


def test_ml1m_double_colon_and_sparse_ids(tmp_path: Path) -> None:
    raw = tmp_path / "ml-1m"
    raw.mkdir()
    (raw / "ratings.dat").write_text(
        "\n".join(
            [
                "1::10::5::100",
                "1::20::4::101",
                "1::30::3::102",
                "2::10::5::100",
                "2::20::4::101",
                "2::3952::5::103",  # discontinuous MovieID
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (raw / "movies.dat").write_text(
        "\n".join(
            [
                "10::Toy Story (1995)::Animation",
                "20::Jumanji (1995)::Adventure",
                "30::Grumpier Old Men (1995)::Comedy",
                "3952::Contender, The (2000)::Drama",
            ]
        )
        + "\n",
        encoding="ISO-8859-1",
    )
    (raw / "users.dat").write_text("1::M::25::1::48067\n", encoding="utf-8")

    cfg = MLLM4RecDataConfig(
        dataset_code="ml-1m",
        raw_dir=raw,
        output_root=tmp_path / "out",
        min_uc=2,
        min_sc=1,
        save_parquet=False,
    )
    ds = ML1MDataset(cfg)
    df = ds.load_ratings_df()
    assert set(df["sid"]) == {10, 20, 30, 3952}
    meta = ds.load_meta_dict()
    assert 3952 in meta
    assert meta[3952].endswith("(2000)")
