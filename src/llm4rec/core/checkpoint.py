"""Checkpoint helpers shared by trainers and ExperimentManager."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CheckpointManager:
    """Manage stage checkpoints under ``<run_dir>/checkpoints/``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, stage: str) -> Path:
        path = self.root / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def final_adapter_dir(self, stage: str) -> Path:
        path = self.stage_dir(stage) / "final"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_summary(self, stage: str, summary: dict[str, Any]) -> Path:
        path = self.stage_dir(stage) / "summary.json"
        path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    def latest_checkpoint(self, stage: str) -> Path | None:
        stage_path = self.stage_dir(stage)
        final = stage_path / "final"
        if final.is_dir() and any(final.iterdir()):
            return final
        ckpts = sorted(
            stage_path.glob("checkpoint-*"),
            key=lambda p: p.stat().st_mtime,
        )
        return ckpts[-1] if ckpts else None

    def list_stages(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())
