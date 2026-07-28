"""Seed stability for candidate construction."""

from __future__ import annotations

from pathlib import Path

from llm4rec_bias_Integrated.core.schemas import TaskSpec
from llm4rec_bias_Integrated.datasets.movielens.ml100k import MovieLens100KAdapter


def _tiny_adapter(tmp_path: Path) -> MovieLens100KAdapter:
    raw = tmp_path / "ml-100k"
    raw.mkdir(parents=True)
    genres = "0|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0"
    item_lines = [
        f"{i}|Movie {i} (1995)|01-Jan-1995||http://x|{genres}" for i in range(1, 12)
    ]
    (raw / "u.item").write_text("\n".join(item_lines) + "\n", encoding="latin-1")
    lines: list[str] = []
    for t, item in enumerate([1, 2, 3, 4, 5, 6], start=1):
        lines.append(f"1\t{item}\t5\t{1000 + t}")
    for t, item in enumerate([7, 8, 9, 10, 11, 1], start=1):
        lines.append(f"2\t{item}\t5\t{2000 + t}")
    (raw / "u.data").write_text("\n".join(lines) + "\n", encoding="utf-8")

    adapter = MovieLens100KAdapter(
        data_root=tmp_path / "data",
        min_user_interactions=5,
        seed=123,
        rating_threshold=4.0,
    )
    adapter.raw_dir = tmp_path
    adapter.processed_dir = tmp_path / "processed"
    adapter.cache_dir = tmp_path / "cache"
    adapter._ensure_raw = lambda: raw  # type: ignore[method-assign]
    adapter.preprocess()
    return adapter


def test_same_seed_same_candidates(tmp_path: Path) -> None:
    adapter = _tiny_adapter(tmp_path)
    spec = TaskSpec(
        task="candidate_choice",
        history_max_length=5,
        candidate_size=4,
        negative_sampling="uniform",
        target_position="random",
        framing="neutral",
    )
    a = adapter.build_examples("test", spec)
    b = adapter.build_examples("test", spec)
    assert [ex.candidates for ex in a] == [ex.candidates for ex in b]
    assert [ex.target_index for ex in a] == [ex.target_index for ex in b]
