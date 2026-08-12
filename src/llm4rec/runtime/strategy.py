"""Distributed strategy resolver (DDP default; FSDP/ZeRO optional)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from llm4rec.runtime.hardware import HardwareInfo
from llm4rec.runtime.memory_estimate import estimate_static_training_bytes


@dataclass
class StrategyChoice:
    strategy: str  # single | ddp | fsdp | deepspeed_zero2 | deepspeed_zero3
    backend: str  # nccl | gloo | none
    source: str
    requested_strategy: str = "auto"
    resolved_strategy: str = ""
    effective_strategy: str = ""
    fallback_reason: str | None = None
    model_params_b: float | None = None
    memory_estimate: dict[str, Any] = field(default_factory=dict)
    pressure_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _auto_thresholds(hw_cfg: dict[str, Any] | None) -> tuple[float, float]:
    block = (hw_cfg or {}).get("strategy_auto") or {}
    ddp_t = float(block.get("ddp_pressure_threshold") or 0.45)
    fsdp_t = float(block.get("fsdp_pressure_threshold") or 0.70)
    return ddp_t, fsdp_t


def resolve_strategy(
    requested: str | None,
    hw: HardwareInfo,
    *,
    route: str = "",
    mode: str = "integrated",
    model_params_b: float | None = None,
    deepspeed: str | None = None,
    precision: str = "bf16",
    has_reference_model: bool = False,
    optimizer: str = "adamw",
    stage: str = "sft",
    hw_cfg: dict[str, Any] | None = None,
    free_vram_bytes: int | None = None,
) -> StrategyChoice:
    """Choose distributed strategy.

    Depends on GPU count, free VRAM pressure, model size, route, stage, and
    overrides — never on GPU marketing names.
    """
    req_raw = requested if requested is not None else "auto"
    req = str(req_raw).lower().strip()
    backend = "nccl" if hw.cuda_available else ("gloo" if hw.world_size > 1 else "none")
    free = free_vram_bytes if free_vram_bytes is not None else hw.free_memory
    mem_est: dict[str, Any] = {}
    pressure: float | None = None
    if model_params_b is not None and model_params_b > 0:
        est = estimate_static_training_bytes(
            int(model_params_b * 1e9),
            precision=precision,
            optimizer=optimizer,
            has_reference_model=has_reference_model,
            free_vram_bytes=free,
        )
        mem_est = est.to_dict()
        pressure = est.pressure_ratio

    def _choice(strat: str, source: str, *, fallback: str | None = None) -> StrategyChoice:
        return StrategyChoice(
            strategy=strat,
            backend=backend,
            source=source,
            requested_strategy=str(req_raw),
            resolved_strategy=strat,
            effective_strategy=strat,
            fallback_reason=fallback,
            model_params_b=model_params_b,
            memory_estimate=mem_est,
            pressure_ratio=pressure,
        )

    if deepspeed and str(deepspeed).lower() not in {"null", "none", ""}:
        name = str(deepspeed).lower()
        strat = f"deepspeed_{name}" if not name.startswith("deepspeed") else name
        return _choice(strat, "deepspeed_override")

    if req not in {
        "auto",
        "single",
        "ddp",
        "fsdp",
        "zero2",
        "zero3",
        "deepspeed_zero2",
        "deepspeed_zero3",
    }:
        raise ValueError(f"unsupported strategy '{requested}'")

    if req != "auto":
        mapping = {
            "single": "single",
            "ddp": "ddp",
            "fsdp": "fsdp",
            "zero2": "deepspeed_zero2",
            "zero3": "deepspeed_zero3",
            "deepspeed_zero2": "deepspeed_zero2",
            "deepspeed_zero3": "deepspeed_zero3",
        }
        return _choice(mapping[req], "override")

    if hw.world_size <= 1 or hw.device_count <= 1:
        return _choice("single", "auto")

    ddp_t, fsdp_t = _auto_thresholds(hw_cfg)
    stage_l = str(stage or "sft").lower()
    # Custom trainers cannot execute DeepSpeed yet — never auto-select it for them.
    custom_trainer = stage_l in {"rl", "grpo", "dpo", "sid"}
    allow_deepspeed = stage_l in {"sft", "inference"} and not custom_trainer

    # Memory-pressure path (preferred when free VRAM known)
    if pressure is not None:
        if pressure >= fsdp_t:
            return _choice("fsdp", "auto_memory_pressure")
        if pressure >= ddp_t:
            # Medium pressure: still DDP; callers may shrink microbatch / checkpoint
            return _choice("ddp", "auto_memory_pressure")
        return _choice("ddp", "auto_memory_pressure")

    # Legacy size heuristics when free VRAM unknown
    if allow_deepspeed and mode == "reproduction" and route == "minionerec" and (model_params_b or 0) >= 3.0:
        return _choice("deepspeed_zero2", "auto")

    if mode == "reproduction" and route == "recr1" and (model_params_b or 0) >= 7.0:
        return _choice("fsdp", "auto")

    if (model_params_b or 0) >= 7.0:
        return _choice("fsdp", "auto")

    return _choice("ddp", "auto")


def optional_backend_message(strategy: str) -> str | None:
    """Return an actionable message if an optional backend is missing."""
    if strategy.startswith("deepspeed"):
        try:
            import deepspeed  # noqa: F401
        except ImportError:
            return (
                f"strategy={strategy} requires deepspeed. "
                "Install deepspeed or set hardware.strategy=ddp / hardware.deepspeed=null."
            )
    if strategy == "fsdp":
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel  # noqa: F401
        except ImportError:
            return "FSDP is unavailable in this PyTorch build; use hardware.strategy=ddp."
    return None


def apply_strategy_fallback(
    choice: StrategyChoice,
    hw: HardwareInfo,
    *,
    mode: str,
    log=print,
) -> StrategyChoice:
    """If optional backend missing, fall back to DDP/single and record reason."""
    msg = optional_backend_message(choice.strategy)
    if not msg:
        choice.effective_strategy = choice.resolved_strategy or choice.strategy
        return choice
    fallback = "ddp" if hw.world_size > 1 else "single"
    if mode == "reproduction" or choice.strategy.startswith(("deepspeed", "fsdp")):
        log(f"[runtime] WARNING: {msg}; falling back to {fallback}")
        choice.fallback_reason = msg
        choice.strategy = fallback
        choice.effective_strategy = fallback
        choice.source = "fallback"
        return choice
    log(f"[runtime] WARNING: {msg}")
    choice.fallback_reason = msg
    choice.effective_strategy = choice.strategy
    return choice
