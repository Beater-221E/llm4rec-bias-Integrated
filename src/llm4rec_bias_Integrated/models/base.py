"""Model loading utilities and GPU-only dtype / device selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from llm4rec_bias_Integrated.core.exceptions import ConfigurationError


@dataclass(frozen=True)
class PrecisionConfig:
    """Resolved dtype and AMP flags for the current CUDA device."""

    dtype: torch.dtype
    bf16: bool
    fp16: bool
    device: str  # always "cuda" in this lab


def require_cuda() -> None:
    """Fail fast when CUDA is unavailable or kernels cannot execute."""
    if not torch.cuda.is_available():
        raise ConfigurationError(
            "CUDA is required for llm4rec-bias-Integrated training/eval. "
            "No GPU was detected (torch.cuda.is_available() is False)."
        )
    try:
        x = torch.zeros(1, device="cuda")
        _ = (x + 1).item()
    except Exception as exc:  # noqa: BLE001
        raise ConfigurationError(
            "CUDA is visible but a probe kernel failed. "
            "Install a PyTorch build that includes this GPU's compute capability "
            f"(e.g. cu126 for V100). Underlying error: {exc}"
        ) from exc


def select_device() -> str:
    """Return ``cuda`` or raise. CPU/MPS fallbacks are intentionally unsupported."""
    require_cuda()
    return "cuda"


def resolve_precision(requested: str | None = None) -> PrecisionConfig:
    """Pick model dtype + AMP flags on CUDA.

    ``requested`` may be float16 / bfloat16 / float32 / auto.
    Pre-Ampere GPUs (CC < 8.0, e.g. V100) never use bf16, even if
    ``torch.cuda.is_bf16_supported()`` returns True.
    """
    require_cuda()
    req = (requested or "auto").lower().replace("torch.", "")
    major, minor = torch.cuda.get_device_capability()
    bf16_ok = bool(torch.cuda.is_bf16_supported()) and major >= 8

    if req in {"bfloat16", "bf16"}:
        if not bf16_ok:
            raise ConfigurationError(
                "dtype=bfloat16 requested but this GPU does not support reliable bf16 "
                f"(capability={major}.{minor}). Use float16."
            )
        return PrecisionConfig(torch.bfloat16, True, False, "cuda")
    if req in {"float16", "fp16", "half"}:
        return PrecisionConfig(torch.float16, False, True, "cuda")
    if req in {"float32", "fp32"}:
        return PrecisionConfig(torch.float32, False, False, "cuda")
    # auto
    if bf16_ok:
        return PrecisionConfig(torch.bfloat16, True, False, "cuda")
    return PrecisionConfig(torch.float16, False, True, "cuda")


def count_parameters(model: torch.nn.Module) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_pct": float(100.0 * trainable / total) if total else 0.0,
    }


def hardware_preflight(model_name: str, precision: PrecisionConfig) -> dict[str, Any]:
    """Collect a startup preflight report (requires working CUDA)."""
    require_cuda()
    info: dict[str, Any] = {
        "model": model_name,
        "device": "cuda",
        "dtype": str(precision.dtype).replace("torch.", ""),
        "bf16": precision.bf16,
        "fp16": precision.fp16,
        "cuda_available": True,
        "gpu_count": int(torch.cuda.device_count()),
        "gpu_names": [],
        "gpu_memory_gb": [],
        "compute_capability": [],
        "flash_attention_available": False,
        "notes": [],
    }
    for i in range(torch.cuda.device_count()):
        info["gpu_names"].append(torch.cuda.get_device_name(i))
        info["compute_capability"].append(list(torch.cuda.get_device_capability(i)))
        props = torch.cuda.get_device_properties(i)
        info["gpu_memory_gb"].append(round(props.total_memory / (1024**3), 2))
    try:
        import flash_attn  # noqa: F401

        info["flash_attention_available"] = True
    except Exception:
        info["flash_attention_available"] = False
        info["notes"].append("FlashAttention not installed; using SDPA/eager attention.")
    return info
