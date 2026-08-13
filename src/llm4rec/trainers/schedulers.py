"""Learning-rate schedulers for custom GRPO / DPO loops.

Advance on **optimizer steps**, not micro-batch / accumulation steps.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_type: str = "constant",
    num_training_steps: int,
    warmup_ratio: float | None = None,
    warmup_steps: int | None = None,
) -> LambdaLR:
    """Create a LambdaLR matching common HF schedule names.

    Supports ``constant``, ``cosine``, ``linear`` with optional warmup.
    """
    total = max(1, int(num_training_steps))
    if warmup_steps is not None and warmup_steps not in (False, "null"):
        warm = max(0, int(warmup_steps))
    elif warmup_ratio is not None and warmup_ratio not in (False, "null"):
        warm = max(0, int(float(warmup_ratio) * total))
    else:
        warm = 0
    kind = str(scheduler_type or "constant").lower()

    def lr_lambda(step: int) -> float:
        # step is 0-indexed after each scheduler.step()
        if warm > 0 and step < warm:
            return float(step + 1) / float(max(1, warm))
        if kind in {"constant", "constant_with_warmup"}:
            return 1.0
        progress = float(step - warm) / float(max(1, total - warm))
        progress = min(max(progress, 0.0), 1.0)
        if kind == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if kind == "linear":
            return max(0.0, 1.0 - progress)
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def build_optimizer(
    parameters: Any,
    *,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    optim_name: str | None = None,
) -> tuple[torch.optim.Optimizer, str | None]:
    """Build AdamW; attempt paged AdamW when requested.

    Returns ``(optimizer, fallback_reason_or_None)``.
    """
    name = str(optim_name or "adamw").lower()
    params = list(parameters)
    if name in {"paged_adamw_32bit", "paged_adamw_8bit", "paged_adamw"}:
        try:
            import bitsandbytes as bnb  # type: ignore

            cls = getattr(bnb.optim, "PagedAdamW32bit", None) or getattr(
                bnb.optim, "PagedAdamW", None
            )
            if cls is not None:
                return (
                    cls(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
                    None,
                )
        except Exception:
            pass
        opt = torch.optim.AdamW(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        return opt, f"{name}_unavailable_using_adamw"
    return (
        torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
        None,
    )
