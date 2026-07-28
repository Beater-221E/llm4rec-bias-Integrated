"""Base trainer interface with shared infrastructure hooks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from llm4rec.core.checkpoint import CheckpointManager
from llm4rec.core.context import ExperimentContext
from llm4rec.components.trainer._impl.base import Trainer as LegacyTrainer
from llm4rec.components.trainer._impl.distributed import (
    resolve_distributed_plan,
)


class BaseTrainer(ABC):
    """Shared trainer contract.

    Provides checkpoint / logging / distributed / mixed-precision hooks.
    Workflows define loss, input format, and schedule in subclasses.
    """

    name: str = "base"

    def __init__(self) -> None:
        self._last_summary: dict[str, Any] | None = None

    @abstractmethod
    def train(self, context: ExperimentContext) -> dict[str, Any]:
        ...

    def save(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def checkpoint_manager(self, context: ExperimentContext) -> CheckpointManager:
        return CheckpointManager(context.run_dir / "checkpoints")

    def distributed_plan(self, context: ExperimentContext):
        return resolve_distributed_plan(
            context.config.get("training") or {},
            model_name=context.model_name,
        )

    @staticmethod
    def is_main() -> bool:
        import os

        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


# Keep legacy Trainer importable via components.trainer.base
Trainer = LegacyTrainer

__all__ = ["BaseTrainer", "Trainer", "resolve_distributed_plan"]
