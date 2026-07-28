"""GRPO4Rec trainer adapter (letter SFT / GRPO)."""

from __future__ import annotations

from llm4rec.components.trainer.grpo import GRPOTrainer
from llm4rec.components.trainer.sft import SFTTrainer

__all__ = ["SFTTrainer", "GRPOTrainer"]
