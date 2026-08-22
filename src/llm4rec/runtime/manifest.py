"""Write auditable ``execution_manifest.yaml`` for each run."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _bytes_to_gb(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    # Heuristic: values > 1024 are bytes
    if v > 1024:
        return round(v / (1024**3), 3)
    return round(v, 3)


def build_execution_manifest(
    cfg: dict[str, Any],
    *,
    runtime: Any = None,
    batch_plans: dict[str, Any] | None = None,
    throughput: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hw = cfg.get("hardware") or {}
    sid = cfg.get("sid") or {}
    train = cfg.get("train") or {}
    sft = train.get("sft") or {}
    rl = train.get("rl") or {}
    grpo = rl.get("grpo") or {}
    opt = cfg.get("optimization") or {}
    gen = opt.get("generation") or {}

    hw_info = {}
    if runtime is not None and hasattr(runtime, "hardware"):
        hw_info = runtime.hardware.to_dict()
    elif isinstance(hw.get("_hardware"), dict):
        hw_info = dict(hw["_hardware"])

    snap = runtime.execution_snapshot() if runtime is not None and hasattr(runtime, "execution_snapshot") else {}

    # Prefer primary stage batch plan for flat execution fields
    primary_plan: dict[str, Any] = {}
    for key in ("rl", "sft", "dpo", "distill"):
        if batch_plans and key in batch_plans:
            primary_plan = batch_plans[key] or {}
            break
    if not primary_plan and batch_plans:
        primary_plan = next(iter(batch_plans.values()), {}) or {}

    algo = {
        "mode": cfg.get("mode"),
        "route": (cfg.get("experiment") or {}).get("route"),
        "stages": list(cfg.get("stages") or []),
        "seed": cfg.get("seed"),
        "sid": {
            "implementation": sid.get("implementation"),
            "method": sid.get("method"),
            "rqvae_seed": (sid.get("rqvae") or {}).get("seed"),
            "levels": sid.get("levels"),
            "codebook_size": sid.get("codebook_size"),
        },
    }

    reproduction_scope = {
        "method": (cfg.get("experiment") or {}).get("route") or "minionerec",
        "algorithm_semantics": "reference"
        if str(cfg.get("mode") or "").lower() == "reproduction"
        else "integrated",
        "data_protocol": "integrated_unified",
        "algorithm": "minionerec_reference"
        if str(cfg.get("mode") or "").lower() == "reproduction"
        and str((cfg.get("experiment") or {}).get("route") or "") == "minionerec"
        else None,
    }
    # Drop nulls
    reproduction_scope = {k: v for k, v in reproduction_scope.items() if v is not None}

    hardware_block = {
        "gpu_count": hw_info.get("device_count") or hw_info.get("world_size"),
        "gpu_name": hw_info.get("gpu_name") or hw_info.get("device_name"),
        "compute_capability": hw_info.get("compute_capability"),
        "total_vram_gb": _bytes_to_gb(hw_info.get("total_memory") or hw_info.get("total_vram_gb")),
        "free_vram_at_start_gb": _bytes_to_gb(hw_info.get("free_memory") or hw_info.get("free_vram_gb")),
        "raw": hw_info,
    }

    act_block = hw.get("_activation_checkpointing")
    if not isinstance(act_block, dict):
        act_block = {
            "requested": hw.get("activation_checkpointing"),
            "effective": hw.get("_activation_checkpointing_effective", hw.get("activation_checkpointing")),
            "reason": None,
        }

    kv = gen.get("_cache_choice") or {
        "requested": gen.get("cache"),
        "effective": gen.get("_effective_cache") or gen.get("cache"),
        "fallback_reason": gen.get("_cache_fallback_reason"),
    }

    model_cfg = cfg.get("model") or {}
    execution = {
        "requested_strategy": snap.get("requested_strategy") or hw.get("strategy"),
        "resolved_strategy": snap.get("resolved_strategy")
        or (hw.get("_resolved_strategy") or {}).get("resolved_strategy"),
        "effective_strategy": snap.get("effective_strategy")
        or (hw.get("_resolved_strategy") or {}).get("effective_strategy"),
        "fallback_reason": snap.get("fallback_reason")
        or (hw.get("_resolved_strategy") or {}).get("fallback_reason"),
        "precision": snap.get("precision") or hw.get("precision"),
        "amp": (snap.get("precision") or {}).get("amp") if isinstance(snap.get("precision"), dict) else None,
        "grad_scaler": (snap.get("precision") or {}).get("grad_scaler")
        if isinstance(snap.get("precision"), dict)
        else None,
        "activation_checkpointing": act_block,
        "per_device_batch": primary_plan.get("per_device_batch_size"),
        "gradient_accumulation": primary_plan.get("gradient_accumulation_steps"),
        "target_global_batch": primary_plan.get("target_global_batch_size")
        or primary_plan.get("reference_global_batch"),
        "effective_global_batch": primary_plan.get("effective_global_batch_size"),
        "batch_deviation": primary_plan.get("relative_batch_deviation"),
        "compile_requested": snap.get("compile_requested"),
        "compile_effective": snap.get("compile_effective"),
        "compile_backend": snap.get("compile_backend"),
        "compile_mode": snap.get("compile_mode"),
        "compile_fallback_reason": snap.get("compile_fallback_reason"),
        "attention_backend": snap.get("attention_implementation"),
        "kv_cache": kv,
        "batch_plans": batch_plans or {},
        "model_params_b": snap.get("model_params_b") or hw.get("_model_params_b"),
        "world_size": snap.get("world_size"),
        "memory_estimate": hw.get("_memory_estimate"),
        "sid_token_initialization": model_cfg.get("_sid_token_initialization_effective")
        or model_cfg.get("sid_token_initialization"),
    }

    # Measured performance only (no dummy zeros)
    performance: dict[str, Any] = {}
    for key in ("sft", "grpo", "dpo", "sid", "distill", "transition"):
        block = (cfg.get("_performance") or {}).get(key)
        if isinstance(block, dict) and block:
            performance[key] = block
    if throughput:
        # Merge any explicitly passed throughput that has real values
        for k, v in throughput.items():
            if v is None:
                continue
            if isinstance(v, dict) and not v:
                continue
            performance.setdefault("extra", {})[k] = v

    reference_semantics = {
        "sft_objectives": sft.get("objectives") or sft.get("tasks"),
        "optimizer": rl.get("optim") or rl.get("optimizer") or "adamw",
        "scheduler": rl.get("lr_scheduler_type"),
        "warmup": rl.get("warmup_ratio") if rl.get("warmup_ratio") is not None else rl.get("warmup_steps"),
        "sft_lr": sft.get("learning_rate"),
        "sft_epochs": sft.get("epochs"),
        "sft_scheduler": sft.get("lr_scheduler_type"),
        "rollout_mode": "constrained_beam" if grpo.get("constrained_rollout", True) else "sample",
        "num_generations": grpo.get("group_size"),
        "temperature": grpo.get("temperature"),
        "beam_search": grpo.get("beam_search", True),
        "do_sample": grpo.get("do_sample"),
        "ref_sync": {
            "enabled": grpo.get("sync_ref_model"),
            "alpha": grpo.get("ref_model_mixup_alpha"),
            "steps": grpo.get("ref_model_sync_steps"),
        },
    }

    return {
        "algorithm": algo,
        "reproduction_scope": reproduction_scope,
        "reference": cfg.get("reference") or {},
        "hardware": hardware_block,
        "execution": execution,
        "reference_semantics": reference_semantics,
        "performance": performance,
        "model": {
            "sid_token_initialization": execution.get("sid_token_initialization"),
        },
    }


def write_execution_manifest(path: Path | str, manifest: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        out.write_text(
            yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except Exception:
        import json

        out.with_suffix(".json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return out.with_suffix(".json")
    return out
