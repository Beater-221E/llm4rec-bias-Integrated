"""File-based eval gather: no NCCL, wait on shard .done markers."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm4rec.eval.bias import RankedResult
from llm4rec.eval.gather import gather_ranked_results


def _row(user: str, item: str) -> RankedResult:
    return RankedResult(
        user_id=user,
        ranked_items=[item, "x"],
        target_item=item,
        history=["h"],
        valid=True,
    )


def test_ranked_result_roundtrip():
    src = _row("u1", "i1")
    got = RankedResult.from_dict(src.to_dict())
    assert got == src


def test_gather_single_process_passthrough(tmp_path: Path):
    local = [_row("u0", "a")]
    out = gather_ranked_results(local, tmp_path, name="eval", world_size=1, rank=0)
    assert out == local
    assert list(tmp_path.glob("*")) == []


def test_gather_two_ranks_via_files(tmp_path: Path):
    from concurrent.futures import ThreadPoolExecutor

    a = [_row("u0", "a")]
    b = [_row("u1", "b"), _row("u2", "c")]
    kwargs = dict(name="eval", world_size=2, poll_s=0.01, timeout_s=2.0)
    with ThreadPoolExecutor(2) as pool:
        f0 = pool.submit(gather_ranked_results, a, tmp_path, rank=0, **kwargs)
        f1 = pool.submit(gather_ranked_results, b, tmp_path, rank=1, **kwargs)
        out0, out1 = f0.result(), f1.result()
    assert [r.user_id for r in out0] == ["u0", "u1", "u2"]
    assert out0 == out1
    assert (tmp_path / "eval.rank0.done").is_file()
    assert (tmp_path / "eval.rank1.jsonl").is_file()


def test_gather_timeout_lists_missing_rank(tmp_path: Path):
    with pytest.raises(TimeoutError, match="eval.rank1.done"):
        gather_ranked_results(
            [_row("u0", "a")],
            tmp_path,
            name="eval",
            rank=0,
            world_size=2,
            timeout_s=0.05,
            poll_s=0.01,
        )
