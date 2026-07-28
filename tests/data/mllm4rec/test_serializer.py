"""Serializer / schema tests."""

from __future__ import annotations

from pathlib import Path

from llm4rec_bias_Integrated.data.mllm4rec.compatibility import (
    convert_to_llm4rec_bias_schema,
    validate_official_schema,
)
from llm4rec_bias_Integrated.data.mllm4rec.serializer import load_pickle, save_pickle


def _toy_dataset():
    return {
        "train": {1: [1, 2]},
        "val": {1: [3]},
        "test": {1: [4]},
        "meta": {1: "A (1995)", 2: "B (1995)", 3: "C (1995)", 4: "D (1995)"},
        "umap": {10: 1},
        "smap": {100: 1, 101: 2, 102: 3, 103: 4},
    }


def test_atomic_pickle_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "dataset.pkl"
    data = _toy_dataset()
    save_pickle(data, path, atomic_write=True, create_backup=False)
    loaded = load_pickle(path)
    assert loaded["meta"][1] == "A (1995)"
    assert validate_official_schema(loaded) == []


def test_backup_on_rewrite(tmp_path: Path) -> None:
    path = tmp_path / "dataset.pkl"
    save_pickle(_toy_dataset(), path, atomic_write=True, create_backup=True)
    data2 = _toy_dataset()
    data2["meta"][1] = "Changed (1995)"
    save_pickle(data2, path, atomic_write=True, create_backup=True)
    assert path.with_suffix(".pkl.bak").is_file()
    bak = load_pickle(path.with_suffix(".pkl.bak"))
    assert bak["meta"][1] == "A (1995)"


def test_convert_does_not_clobber_meta() -> None:
    data = _toy_dataset()
    converted = convert_to_llm4rec_bias_schema(data)
    assert converted["meta"] == data["meta"]
    assert "extended_metadata" in converted
