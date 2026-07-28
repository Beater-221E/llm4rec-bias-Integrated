"""Shared GRPO trainer facade."""

from __future__ import annotations

from typing import Any

from llm4rec.components.trainer.base import BaseTrainer
from llm4rec.components.trainer._impl.grpo import GRPOLoRATrainer
from llm4rec.core.context import ExperimentContext
from llm4rec.core.schemas import RecommendationExample


class GRPOTrainer(BaseTrainer):
    """GRPO stage wrapper; reward composition comes from config plugins."""

    name = "grpo"

    def __init__(
        self,
        train_examples: list[RecommendationExample] | None = None,
        *,
        sft_adapter_path: str | None = None,
        impl: GRPOLoRATrainer | None = None,
    ) -> None:
        super().__init__()
        if impl is not None:
            self._impl = impl
        else:
            self._impl = GRPOLoRATrainer(
                train_examples or [],
                sft_adapter_path=sft_adapter_path,
            )

    def train(self, context: ExperimentContext) -> dict[str, Any]:
        summary = self._impl.train(context)
        self._last_summary = summary
        return summary

    def save(self, output_dir):
        return self._impl.save(output_dir)


__all__ = ["GRPOTrainer", "GRPOLoRATrainer"]
