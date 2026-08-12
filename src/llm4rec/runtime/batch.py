"""Preserve target global batch across hardware with best-effort deviation control."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from llm4rec.core.exceptions import ConfigurationError


@dataclass
class BatchPlan:
    world_size: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    effective_global_batch_size: int
    target_global_batch_size: int | None
    reference_global_batch: int | None
    relative_batch_deviation: float
    adjusted: bool
    preserve_policy: str
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_lines(self) -> list[str]:
        return [
            f"world_size                    : {self.world_size}",
            f"per_device_batch_size         : {self.per_device_batch_size}",
            f"gradient_accumulation_steps  : {self.gradient_accumulation_steps}",
            f"reference_global_batch        : {self.reference_global_batch}",
            f"effective_global_batch_size  : {self.effective_global_batch_size}",
            f"relative_batch_deviation     : {self.relative_batch_deviation:.4f}",
            f"preserve_policy              : {self.preserve_policy}",
        ]


def _policy_from_cfg(batch_policy: dict[str, Any] | None) -> tuple[str, float]:
    bp = batch_policy or {}
    preserve = str(bp.get("preserve_global_batch") or "best_effort").lower()
    if preserve not in {"best_effort", "strict"}:
        preserve = "best_effort"
    if "max_relative_deviation" in bp and bp["max_relative_deviation"] is not None:
        max_dev = float(bp["max_relative_deviation"])
    else:
        max_dev = 0.05
    return preserve, max_dev


def resolve_batch_plan(
    *,
    world_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int | None = None,
    global_batch_size: int | None = None,
    target_global_batch_size: int | None = None,
    mode: str = "integrated",
    memory_auto: bool = False,
    preferred_per_device: int | None = None,
    batch_policy: dict[str, Any] | None = None,
) -> BatchPlan:
    """Compute micro-batch / accumulation for a target global batch.

    Default policy is ``best_effort`` on all modes (including reproduction).
    ``strict`` raises when relative deviation exceeds ``max_relative_deviation``.
    """
    _ = mode  # mode no longer forces hard-fail; use batch_policy instead
    ws = max(1, int(world_size))
    micro = max(1, int(per_device_batch_size))
    preserve, max_dev = _policy_from_cfg(batch_policy)

    if preferred_per_device is not None and memory_auto:
        micro = max(1, int(preferred_per_device))

    raw_target = target_global_batch_size if target_global_batch_size not in (None, 0, "null") else global_batch_size
    target = int(raw_target) if raw_target not in (None, 0, "null") else None

    if target is not None:
        denom = micro * ws
        if target % denom == 0:
            accum = target // denom
            effective = target
            adjusted = False
            msg = ""
            rel = 0.0
        else:
            floor_accum = max(1, target // denom)
            ceil_accum = max(1, math.ceil(target / denom))
            floor_eff = micro * ws * floor_accum
            ceil_eff = micro * ws * ceil_accum
            if abs(ceil_eff - target) < abs(floor_eff - target):
                accum, effective = ceil_accum, ceil_eff
            else:
                accum, effective = floor_accum, floor_eff
            adjusted = effective != target
            rel = abs(effective - target) / max(target, 1)
            msg = (
                f"target_global_batch={target} not exact with "
                f"per_device({micro})*world_size({ws}); "
                f"using accum={accum} → effective={effective} "
                f"(rel_dev={rel:.4f}, policy={preserve})"
            )
            if preserve == "strict" and rel > max_dev + 1e-12:
                raise ConfigurationError(
                    f"[batch_policy=strict] cannot preserve global_batch within "
                    f"{max_dev:.2%}: {msg}"
                )
        return BatchPlan(
            world_size=ws,
            per_device_batch_size=micro,
            gradient_accumulation_steps=max(1, int(accum)),
            effective_global_batch_size=int(effective),
            target_global_batch_size=target,
            reference_global_batch=target,
            relative_batch_deviation=float(rel) if adjusted else 0.0,
            adjusted=adjusted,
            preserve_policy=preserve,
            message=msg,
        )

    accum = max(1, int(gradient_accumulation_steps or 1))
    effective = micro * ws * accum
    return BatchPlan(
        world_size=ws,
        per_device_batch_size=micro,
        gradient_accumulation_steps=accum,
        effective_global_batch_size=effective,
        target_global_batch_size=None,
        reference_global_batch=None,
        relative_batch_deviation=0.0,
        adjusted=False,
        preserve_policy=preserve,
        message="",
    )


def probe_max_micro_batch(
    *,
    preferred: int,
    min_batch: int = 1,
    probe_fn=None,
    log=print,
) -> int:
    """Decrease micro-batch on OOM: preferred → … → 1 (halving)."""
    if probe_fn is None:
        return preferred
    size = max(min_batch, int(preferred))
    while size >= min_batch:
        try:
            probe_fn(size)
            return size
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            log(f"[memory-auto] OOM at per_device_batch_size={size}; retrying smaller")
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            if size == min_batch:
                raise
            size = max(min_batch, size // 2)
    return min_batch
