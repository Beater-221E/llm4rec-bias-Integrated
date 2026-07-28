"""Optional quantization helpers (QLoRA)."""

from __future__ import annotations

from typing import Any


def build_bnb_config(quant_cfg: dict[str, Any] | None) -> Any | None:
    """Return BitsAndBytesConfig when 4-bit is requested, else None."""
    if not quant_cfg or not quant_cfg.get("load_in_4bit"):
        return None
    from transformers import BitsAndBytesConfig
    import torch

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(quant_cfg.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(quant_cfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=getattr(
            torch, str(quant_cfg.get("bnb_4bit_compute_dtype", "float16"))
        ),
    )
