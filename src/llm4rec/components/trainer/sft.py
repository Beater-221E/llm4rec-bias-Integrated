"""Shared SFT trainer facade."""

from __future__ import annotations

from typing import Any

from llm4rec.components.trainer.base import BaseTrainer
from llm4rec.components.trainer._impl.sft import SFTLoRATrainer
from llm4rec.core.context import ExperimentContext
from llm4rec.core.schemas import RecommendationExample


class SFTTrainer(BaseTrainer):
    """SFT stage wrapper around the letter-route LoRA trainer.

    SID / MLLM workflows may subclass or construct their own trainers while
    still sharing checkpoint / distributed helpers from BaseTrainer.
    """

    name = "sft"

    def __init__(
        self,
        train_examples: list[RecommendationExample] | None = None,
        eval_examples: list[RecommendationExample] | None = None,
        *,
        impl: SFTLoRATrainer | None = None,
    ) -> None:
        super().__init__()
        if impl is not None:
            self._impl = impl
        else:
            self._impl = SFTLoRATrainer(train_examples or [], eval_examples)

    def train(self, context: ExperimentContext) -> dict[str, Any]:
        summary = self._impl.train(context)
        self._last_summary = summary
        return summary

    def save(self, output_dir):
        return self._impl.save(output_dir)


__all__ = ["SFTTrainer", "SFTLoRATrainer"]
