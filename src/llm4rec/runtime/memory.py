"""Lightweight memory-aware micro-batch probing."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.distributed as dist

from llm4rec.core import distributed as dist_utils
from llm4rec.runtime.batch import probe_max_micro_batch, resolve_batch_plan


def _broadcast_int(value: int, *, src: int = 0) -> int:
    """Broadcast an int from ``src`` so every rank shares the same micro-batch."""
    if not dist_utils.is_distributed():
        return int(value)
    device = f"cuda:{dist_utils.local_rank()}" if torch.cuda.is_available() else "cpu"
    tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
    dist.broadcast(tensor, src=src)
    return int(tensor.item())


def auto_tune_micro_batch(
    *,
    preferred: int,
    world_size: int,
    global_batch_size: int | None,
    mode: str,
    memory_auto: bool,
    probe_fn: Callable[[int], None] | None = None,
    batch_policy: dict[str, Any] | None = None,
    log=print,
) -> tuple[int, int]:
    """Return ``(per_device_batch_size, gradient_accumulation_steps)``.

    Probe runs on rank0 only (CUDA OOM probe is expensive / can desync ranks),
    then the chosen micro-batch is broadcast so HF Trainer DDP sees identical
    ``per_device_batch_size`` / ``gradient_accumulation_steps`` everywhere.
    """
    micro = int(preferred)
    if memory_auto and probe_fn is not None:
        dist_utils.barrier("memory-auto-before-probe")
        if dist_utils.is_main():
            micro = probe_max_micro_batch(preferred=preferred, probe_fn=probe_fn, log=log)
            if micro != preferred:
                log(f"[memory-auto] per_device_batch_size {preferred} → {micro}")
        micro = _broadcast_int(micro, src=0)
        dist_utils.barrier("memory-auto-after-probe")
        if not dist_utils.is_main():
            log(f"[memory-auto] using rank0 micro-batch={micro}")

    plan = resolve_batch_plan(
        world_size=world_size,
        per_device_batch_size=micro,
        global_batch_size=global_batch_size,
        mode=mode,
        memory_auto=memory_auto,
        preferred_per_device=micro,
        batch_policy=batch_policy,
    )
    if plan.message:
        log(f"[memory-auto] {plan.message}")
    return plan.per_device_batch_size, plan.gradient_accumulation_steps
