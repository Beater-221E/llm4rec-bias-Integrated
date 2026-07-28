"""PEFT / LoRA helpers."""

from __future__ import annotations

from typing import Any

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import PreTrainedModel


def build_lora_config(
    peft_cfg: dict[str, Any],
    *,
    trainable_token_indices: dict[str, list[int]] | None = None,
) -> LoraConfig:
    targets = peft_cfg.get("target_modules") or [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
    kwargs: dict[str, Any] = {
        "r": int(peft_cfg.get("rank", peft_cfg.get("r", 16))),
        "lora_alpha": int(peft_cfg.get("alpha", peft_cfg.get("lora_alpha", 32))),
        "lora_dropout": float(peft_cfg.get("dropout", peft_cfg.get("lora_dropout", 0.05))),
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": list(targets),
    }
    if trainable_token_indices:
        kwargs["trainable_token_indices"] = trainable_token_indices
    return LoraConfig(**kwargs)


def apply_lora(model: PreTrainedModel, peft_cfg: dict[str, Any]) -> PreTrainedModel:
    if not peft_cfg.get("enabled", True):
        return model
    return get_peft_model(model, build_lora_config(peft_cfg))


def load_and_merge_adapter(
    model: PreTrainedModel,
    adapter_path: str,
) -> PreTrainedModel:
    return PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
