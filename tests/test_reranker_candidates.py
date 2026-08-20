"""KAR / DPO4Rec candidate lists: positives + uniform unobserved negatives."""

from __future__ import annotations

import math
import random
from typing import Any

from llm4rec.rerankers.service import RerankerService, _ndcg_at_k
from llm4rec.tracking.progress import overwrite_progress


def _service(n_items: int = 40, candidate_size: int = 10, n_positives: int = 4) -> RerankerService:
    item_ids = [f"i{i}" for i in range(n_items)]
    cfg = {
        "decoder": {
            "reranker": {
                "kind": "prm",
                "candidate_size": candidate_size,
                "n_positives": n_positives,
                "hidden_dim": 16,
                "num_heads": 2,
                "num_layers": 1,
            }
        }
    }
    return RerankerService(cfg, item_ids, device="cpu")


def test_build_candidates_kar_positives_and_unobserved_negatives():
    service = _service()
    example = {
        "target_item": "i0",
        "history": ["i1", "i2", "i3"],
        "positive_items": ["i0", "i4", "i5", "i6"],
    }
    candidates, pos, pos_indices = service.build_candidates(
        example, {}, random.Random(0)
    )
    assert len(candidates) == 10
    assert candidates[pos] == "i0"
    assert {candidates[i] for i in pos_indices} == {"i0", "i4", "i5", "i6"}
    assert not {"i1", "i2", "i3"} & set(candidates)
    assert len(set(candidates)) == 10


def test_build_candidates_single_target_when_no_future():
    service = _service()
    example = {"target_item": "i7", "history": ["i8"]}
    candidates, pos, pos_indices = service.build_candidates(
        example, {}, random.Random(1)
    )
    assert candidates[pos] == "i7"
    assert pos_indices == [pos]
    assert "i8" not in candidates
    assert len(candidates) == 10


def test_assign_candidates_reuses_existing_lists():
    service = _service()
    frozen = [f"i{i}" for i in range(10)]
    examples = [
        {
            "target_item": "i0",
            "history": ["i20"],
            "_candidates": list(frozen),
            "_target_pos": 0,
            "_pos_indices": [0, 1, 2, 3],
        },
        {"target_item": "i4", "history": ["i5"], "positive_items": ["i4"]},
    ]
    service.assign_candidates(examples, {}, random.Random(1), desc="test")
    assert examples[0]["_candidates"] == frozen
    assert examples[0]["_pos_indices"] == [0, 1, 2, 3]
    assert examples[1]["_candidates"][examples[1]["_target_pos"]] == "i4"
    assert len(examples[1]["_candidates"]) == 10


def test_ndcg_single_relevant_matches_old_formula():
    assert _ndcg_at_k([3, 0, 1], {3}, 5) == 1.0
    assert _ndcg_at_k([0, 3, 1], {3}, 5) == 1.0 / math.log2(3)
    assert _ndcg_at_k([0, 1, 2, 3, 4, 5], {5}, 5) == 0.0


def test_overwrite_progress_can_be_disabled():
    with overwrite_progress(10, "test", enabled=False) as bar:
        assert bar.bar is None
        assert bar.update(3) == 3


def test_progress_logs_are_time_throttled(caplog):
    import logging

    caplog.set_level(logging.INFO, logger="llm4rec.tracking.progress")
    with overwrite_progress(100, "eval", enabled=False, log_interval_s=60.0) as bar:
        for _ in range(20):
            bar.update(1)
    # start + maybe close; never one line per update
    records = [r for r in caplog.records if "[eval]" in r.getMessage()]
    assert 1 <= len(records) <= 3


def test_shared_progress_sums_all_ranks(tmp_path, monkeypatch):
    from llm4rec.core import distributed as dist_utils

    monkeypatch.setattr(dist_utils, "world_size", lambda: 4)
    monkeypatch.setattr(dist_utils, "rank", lambda: 0)
    monkeypatch.setattr(dist_utils, "is_main", lambda: True)

    with overwrite_progress(
        2,
        "eval",
        enabled=False,
        global_total=8,
        progress_dir=tmp_path,
        name="eval1",
    ) as bar:
        bar.update(1)
        (tmp_path / "eval1.rank1").write_text("2", encoding="utf-8")
        (tmp_path / "eval1.rank2").write_text("1", encoding="utf-8")
        (tmp_path / "eval1.rank3").write_text("3", encoding="utf-8")
        assert bar.global_done() == 7


def _example_with_candidates() -> dict[str, Any]:
    return {
        "target_item": "i0",
        "history": ["i1", "i2"],
        "positive_items": ["i0", "i3"],
        "_candidates": ["i0", "i3", "i5", "i7", "i9", "i11", "i13", "i15", "i17", "i19"],
        "_target_pos": 0,
        "_pos_indices": [0, 1],
    }


def _text_seeded_knowledge(texts):
    """Deterministic per-text vector so batch vs single agree."""
    import torch

    vecs = []
    for t in texts:
        rng = torch.Generator().manual_seed(abs(hash(t)) % (2**32))
        vecs.append(torch.randn(64, generator=rng))
    return torch.stack(vecs)


def test_score_reasonings_batched_matches_single(monkeypatch):
    service = _service()
    monkeypatch.setattr(service, "_knowledge_vector", _text_seeded_knowledge)
    example = _example_with_candidates()
    texts = ["reasoning A", "reasoning B", "reasoning C"]
    batched = service.score_reasonings(example, texts, top_k=5)
    single = [service.score_reasoning(example, t, top_k=5) for t in texts]
    assert batched == single
    assert len(batched) == len(texts)
    assert service.score_reasonings(example, [], top_k=5) == []


def test_score_reasonings_missing_candidates_raises():
    service = _service()
    example = {"target_item": "i0"}
    try:
        service.score_reasonings(example, ["a"], top_k=5)
        raise AssertionError("expected MissingArtifactError")
    except Exception as exc:
        from llm4rec.core.exceptions import MissingArtifactError

        assert isinstance(exc, MissingArtifactError)
