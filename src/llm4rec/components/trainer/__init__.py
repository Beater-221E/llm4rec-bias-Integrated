"""Trainer layer: shared SFT / GRPO scaffolding."""

from llm4rec.components.trainer.base import BaseTrainer
from llm4rec.components.trainer.sft import SFTTrainer
from llm4rec.components.trainer.grpo import GRPOTrainer

__all__ = ["BaseTrainer", "SFTTrainer", "GRPOTrainer"]
