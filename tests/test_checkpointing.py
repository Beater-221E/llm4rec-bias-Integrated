"""Unit tests for mid-training checkpoint helpers."""

from __future__ import annotations

from pathlib import Path

from llm4rec.runtime.checkpointing import (
    is_better_metric,
    list_step_checkpoints,
    prune_step_checkpoints,
    resolve_save_steps,
    save_best_enabled,
    should_save_at_step,
)


def test_resolve_save_steps_default_and_disable():
    assert resolve_save_steps({"checkpoint": {"save_steps": 500}}) == 500
    assert resolve_save_steps({"checkpoint": {"save_steps": None}}) is None
    assert resolve_save_steps({"checkpoint": {"save_steps": 0}}) is None
    assert resolve_save_steps({"checkpoint": {"save_steps": False}}) is None


def test_resolve_save_steps_fraction_and_stage_override():
    cfg = {"checkpoint": {"save_steps": 500}}
    assert resolve_save_steps(cfg, {"save_steps": 0.1}, max_steps=1000, as_int=True) == 100
    assert resolve_save_steps(cfg, {"save_steps": 0.1}) == 0.1
    assert resolve_save_steps(cfg, {"save_steps": None}) is None


def test_should_save_and_prune(tmp_path: Path):
    assert should_save_at_step(500, 500)
    assert not should_save_at_step(499, 500)
    assert not should_save_at_step(500, None)

    for step in (100, 200, 300, 400):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "final").mkdir()
    removed = prune_step_checkpoints(tmp_path, 2)
    assert [p.name for p in removed] == ["checkpoint-100", "checkpoint-200"]
    left = [p.name for _, p in list_step_checkpoints(tmp_path)]
    assert left == ["checkpoint-300", "checkpoint-400"]
    assert (tmp_path / "final").is_dir()


def test_best_metric_and_save_best_flag():
    assert is_better_metric(0.4, None)
    assert is_better_metric(0.3, 0.4)
    assert not is_better_metric(0.5, 0.4)
    assert is_better_metric(0.9, 0.4, lower_is_better=False)
    assert save_best_enabled({})
    assert save_best_enabled({"checkpoint": {"save_best": True}})
    assert not save_best_enabled({"checkpoint": {"save_best": False}})
