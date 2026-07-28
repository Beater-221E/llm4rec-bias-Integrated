"""LoRA helpers (thin facade over PEFT implementation)."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedModel

from llm4rec.components.model._impl.peft import (
    apply_lora,
    build_lora_config,
    load_and_merge_adapter,
)


def apply_lora_config(model: PreTrainedModel, peft_cfg: dict[str, Any]) -> PreTrainedModel:
    return apply_lora(model, peft_cfg)


def merge_adapter(model: PreTrainedModel, adapter_path: str) -> PreTrainedModel:
    return load_and_merge_adapter(model, adapter_path)


__all__ = ["apply_lora_config", "build_lora_config", "merge_adapter"]
