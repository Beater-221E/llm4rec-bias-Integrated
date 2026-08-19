"""Activation-checkpointing resolution (memory-pressure driven)."""

from __future__ import annotations

from typing import Any


def resolve_activation_checkpointing(
    hw_cfg: dict[str, Any],
    *,
    preferred_micro: int,
    selected_micro: int,
    pressure_ratio: float | None = None,
    effective_strategy: str | None = None,
    strategy_source: str | None = None,
) -> tuple[bool, str]:
    """Return ``(enable, reason)``.

    Reasons: explicit | memory_pressure | microbatch_reduction | fsdp_memory_path
    | disabled_not_needed | disabled_explicit
    """
    requested = hw_cfg.get("activation_checkpointing")
    grad_flag = hw_cfg.get("gradient_checkpointing")
    if requested is None:
        requested = grad_flag if grad_flag is not None else False
    auto_cfg = hw_cfg.get("activation_checkpointing_auto") or {}
    pressure_t = float(auto_cfg.get("pressure_threshold") or 0.65)
    reduction_t = float(auto_cfg.get("microbatch_reduction_ratio") or 0.5)

    if requested in (True, "true", "True", 1):
        return True, "explicit"
    if requested in (False, "false", "False", 0):
        return False, "disabled_explicit"
    if requested is None:
        return False, "disabled_explicit"
    if str(requested).lower() != "auto":
        return bool(requested), "explicit"
    # auto: still honor an explicit gradient_checkpointing=true from configs
    if grad_flag in (True, "true", "True", 1):
        return True, "explicit_gradient_checkpointing"

    preferred = max(1, int(preferred_micro))
    selected = max(1, int(selected_micro))
    reduction_ratio = selected / preferred

    if reduction_ratio <= reduction_t:
        return True, "microbatch_reduction"
    if pressure_ratio is not None and pressure_ratio >= pressure_t:
        return True, "memory_pressure"
    if (
        str(effective_strategy or "").lower() == "fsdp"
        and str(strategy_source or "").startswith("auto_memory")
    ):
        return True, "fsdp_memory_path"
    return False, "disabled_not_needed"
