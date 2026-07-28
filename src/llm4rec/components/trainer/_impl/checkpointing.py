"""Checkpoint path helpers."""

from __future__ import annotations

from pathlib import Path


def stage_dir(run_dir: Path, stage: str) -> Path:
    path = run_dir / "checkpoints" / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def final_adapter_dir(run_dir: Path, stage: str) -> Path:
    path = stage_dir(run_dir, stage) / "final"
    path.mkdir(parents=True, exist_ok=True)
    return path
