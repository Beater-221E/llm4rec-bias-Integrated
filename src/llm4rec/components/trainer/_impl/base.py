"""Trainer interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from llm4rec.core.context import ExperimentContext


class Trainer(ABC):
    """Optimization stage (SFT, GRPO, ...)."""

    name: str

    @abstractmethod
    def train(self, context: ExperimentContext) -> dict[str, Any]:
        """Run training and return a summary dict (paths, metrics)."""

    @abstractmethod
    def save(self, output_dir: Path) -> Path:
        """Persist the latest checkpoint / adapter."""
