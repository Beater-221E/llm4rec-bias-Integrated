"""Unit tests for MovieLens parsing, splits, sampling, prompts."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from llm4rec_bias_Integrated.core.exceptions import DatasetValidationError
from llm4rec_bias_Integrated.core.schemas import Interaction, TaskSpec
from llm4rec_bias_Integrated.datasets.movielens.common import (
    chronological_sequences,
    popularity_from_train_region,
)
from llm4rec_bias_Integrated.datasets.movielens.metadata import parse_title_year
from llm4rec_bias_Integrated.datasets.movielens.ml100k import MovieLens100KAdapter
from llm4rec_bias_Integrated.datasets.movielens.split import (
    leave_one_out_split,
    validate_example_integrity,
    validate_no_leakage,
)
from llm4rec_bias_Integrated.datasets.sampling import get_sampler
from llm4rec_bias_Integrated.datasets.transforms.candidates import place_target
from llm4rec_bias_Integrated.prompts.candidate_choice import parse_choice


def _synth_interactions(n_users: int = 5, n_items: int = 20, length: int = 8):
    rows: list[Interaction] = []
    ts = 1_000
    for u in range(1, n_users + 1):
        for t in range(length):
            item = (u + t) % n_items + 1
            rows.append(
                Interaction(
                    user_id=str(u),
                    item_id=str(item),
                    rating=5.0,
                    timestamp=ts,
                )
            )
            ts += 1
    return rows


def test_parse_title_year() -> None:
    title, year = parse_title_year("Toy Story (1995)")
    assert title == "Toy Story"
    assert year == 1995


def test_leave_one_out_and_no_leakage() -> None:
    rows = _synth_interactions()
    splits = leave_one_out_split(rows, min_user_interactions=5)
    validate_no_leakage(splits)
    assert len(splits.test) == 5
    assert len(splits.validation) == 5
    assert len(splits.train) == 5 * (8 - 2)


def test_history_must_not_contain_target() -> None:
    with pytest.raises(DatasetValidationError):
        validate_example_integrity(["1", "2"], "2", ["2", "3", "4"])


def test_candidates_must_include_target_once() -> None:
    with pytest.raises(DatasetValidationError):
        validate_example_integrity(["1"], "9", ["2", "3", "4"])
    with pytest.raises(DatasetValidationError):
        validate_example_integrity(["1"], "2", ["2", "2", "3"])


def test_uniform_negative_sampling_reproducible() -> None:
    items = [str(i) for i in range(50)]
    s = get_sampler("uniform", item_ids=items)
    a = s.sample(k=5, exclude={"1"}, rng=random.Random(0))
    b = s.sample(k=5, exclude={"1"}, rng=random.Random(0))
    assert a == b
    assert "1" not in a


def test_popularity_negative_sampling() -> None:
    items = [str(i) for i in range(20)]
    counts = {i: (int(i) + 1) for i in items}
    s = get_sampler("popularity", item_ids=items, counts=counts)
    negs = s.sample(k=5, exclude={"0"}, rng=random.Random(1))
    assert len(negs) == 5
    assert "0" not in negs


def test_place_target_positions() -> None:
    rng = random.Random(0)
    cands, pos = place_target("T", ["a", "b", "c"], candidate_size=4, target_position="first", rng=rng)
    assert pos == 0 and cands[0] == "T"
    cands, pos = place_target("T", ["a", "b", "c"], candidate_size=4, target_position="last", rng=rng)
    assert pos == 3 and cands[3] == "T"
    cands, pos = place_target("T", ["a", "b", "c"], candidate_size=4, target_position="middle", rng=rng)
    assert pos == 2 and cands[2] == "T"


def test_parse_choice_strict() -> None:
    assert parse_choice("B", 10) == 1
    assert parse_choice("Answer: C", 10) == 2
    assert parse_choice("B. Fargo (1996)", 10) == 1
    assert parse_choice("Based on history...", 10) is None
    assert parse_choice("", 10) is None


def test_popularity_from_train_region_excludes_holdout() -> None:
    rows = _synth_interactions(n_users=2, length=6)
    seqs = chronological_sequences(rows)
    counts, _ = popularity_from_train_region(seqs, holdout=2)
    # last two items per user must not inflate counts exclusively from holdout-only items
    assert sum(counts.values()) == 2 * (6 - 2)


def test_ml100k_parser_on_fixture(tmp_path: Path) -> None:
    raw = tmp_path / "ml-100k"
    raw.mkdir(parents=True)
    genres = "0|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0"  # Comedy
    item_lines = [
        f"{i}|Movie {i} (1995)|01-Jan-1995||http://x|{genres}" for i in range(1, 12)
    ]
    (raw / "u.item").write_text("\n".join(item_lines) + "\n", encoding="latin-1")
    lines: list[str] = []
    # user 1: items 1..6
    for t, item in enumerate([1, 2, 3, 4, 5, 6], start=1):
        lines.append(f"1\t{item}\t5\t{1000 + t}")
    # user 2: items 7..11 + 1 (ensures a wide train-region popularity pool)
    for t, item in enumerate([7, 8, 9, 10, 11, 1], start=1):
        lines.append(f"2\t{item}\t5\t{2000 + t}")
    (raw / "u.data").write_text("\n".join(lines) + "\n", encoding="utf-8")

    adapter = MovieLens100KAdapter(
        data_root=tmp_path / "data",
        min_user_interactions=5,
        seed=0,
        rating_threshold=4.0,
    )
    adapter.raw_dir = tmp_path
    adapter.processed_dir = tmp_path / "processed"
    adapter.cache_dir = tmp_path / "cache"

    def _ensure() -> Path:
        return raw

    adapter._ensure_raw = _ensure  # type: ignore[method-assign]
    adapter.preprocess()
    splits = adapter.build_splits()
    validate_no_leakage(splits)
    assert len(splits.test) == 2
    examples = adapter.build_examples(
        "test",
        TaskSpec(
            task="candidate_choice",
            history_max_length=5,
            candidate_size=4,
            negative_sampling="uniform",
            target_position="first",
            framing="neutral",
        ),
    )
    assert len(examples) == 2
    ex = examples[0]
    assert ex.target_index == 0
    assert ex.candidates is not None
    assert ex.target_item_id == ex.candidates[0]
    assert ex.target_item_id not in ex.history_item_ids
    assert adapter.fingerprint() == adapter.fingerprint()
