"""Model utility helpers."""

from __future__ import annotations

from llm4rec.components.model._impl.base import PrecisionConfig, require_cuda, resolve_precision
from llm4rec.components.model._impl.quantization import build_bnb_config
from llm4rec.components.model._impl.tokenizer import load_tokenizer


def resolve_precision_name(name: str) -> str:
    """Normalize dtype aliases used in configs."""
    key = (name or "auto").strip().lower()
    aliases = {
        "fp16": "float16",
        "f16": "float16",
        "half": "float16",
        "bf16": "bfloat16",
        "bfloat": "bfloat16",
        "fp32": "float32",
        "f32": "float32",
        "auto": "auto",
        "float16": "float16",
        "bfloat16": "bfloat16",
        "float32": "float32",
    }
    return aliases.get(key, key)


__all__ = [
    "PrecisionConfig",
    "require_cuda",
    "resolve_precision",
    "resolve_precision_name",
    "build_bnb_config",
    "load_tokenizer",
]
