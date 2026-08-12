"""Startup preflight validation and summary table."""

from __future__ import annotations

from typing import Any

from llm4rec.core.modes import get_mode, verify_minionerec_reproduction
from llm4rec.runtime.batch import BatchPlan
from llm4rec.runtime.context import RuntimeContext, build_runtime
from llm4rec.runtime.hardware import HardwareInfo
from llm4rec.runtime.precision import PrecisionChoice
from llm4rec.runtime.strategy import StrategyChoice


def run_preflight(
    cfg: dict[str, Any],
    *,
    runtime: RuntimeContext | None = None,
    log=print,
) -> dict[str, Any]:
    """Resolve hardware/precision/strategy/batch and print a concise table."""
    rt = runtime or build_runtime(cfg, log=log)
    mode = get_mode(cfg)
    route = str((cfg.get("experiment") or {}).get("route") or "")
    hw = rt.hardware
    precision = rt.precision
    strategy = rt.strategy

    batch_plans: dict[str, BatchPlan] = {}
    train = cfg.get("train") or {}
    for stage in ("sft", "rl", "dpo"):
        block = train.get(stage)
        if not isinstance(block, dict):
            continue
        plan = rt.resolve_stage_batch(stage, block)
        if plan.message:
            log(f"[preflight] {stage}: {plan.message}")
        batch_plans[stage] = plan

    sid_notes: list[str] = []
    if route == "minionerec" and mode == "reproduction":
        sid_notes.extend(verify_minionerec_reproduction(cfg))

    compile_cfg = (cfg.get("optimization") or {}).get("compile") or {}
    attn = (cfg.get("optimization") or {}).get("attention") or {}
    gen = (cfg.get("optimization") or {}).get("generation") or {}
    triton = (cfg.get("optimization") or {}).get("triton") or {}

    info = {
        "route": route,
        "mode": mode,
        "model": str((cfg.get("model") or {}).get("checkpoint") or ""),
        "dataset": f"{(cfg.get('data') or {}).get('name')}/{(cfg.get('data') or {}).get('category')}",
        "hardware": hw,
        "precision": precision,
        "strategy": strategy,
        "batch_plans": batch_plans,
        "sid": cfg.get("sid") or {},
        "sid_notes": sid_notes,
        "compile": compile_cfg,
        "attention": attn,
        "generation": gen,
        "triton": triton,
        "amp": precision.amp,
        "grad_scaler": precision.grad_scaler,
    }
    log(format_preflight_table(info))
    return info


def format_preflight_table(info: dict[str, Any]) -> str:
    hw: HardwareInfo = info["hardware"]
    precision: PrecisionChoice = info["precision"]
    strategy: StrategyChoice = info["strategy"]
    sid = info.get("sid") or {}
    plans: dict[str, BatchPlan] = info.get("batch_plans") or {}
    primary = plans.get("rl") or plans.get("dpo") or plans.get("sft")
    compile_cfg = info.get("compile") or {}
    attn = info.get("attention") or {}
    gen = info.get("generation") or {}
    triton = info.get("triton") or {}

    vram = "n/a"
    if hw.total_memory:
        vram = f"{hw.total_memory / (1024**3):.1f} GiB"
        if hw.free_memory is not None:
            vram += f" (free {hw.free_memory / (1024**3):.1f})"

    cc = "n/a"
    if hw.compute_capability:
        cc = f"{hw.compute_capability[0]}.{hw.compute_capability[1]}"

    rows = [
        ("Route", info.get("route")),
        ("Mode", info.get("mode")),
        ("Model", info.get("model")),
        ("Dataset", info.get("dataset")),
        ("GPU count", hw.device_count if hw.cuda_available else 0),
        ("GPU type", hw.gpu_name),
        ("Compute capability", cc),
        ("Free / total VRAM", vram),
        ("Precision", precision.precision),
        ("AMP enabled", info.get("amp")),
        ("GradScaler enabled", info.get("grad_scaler")),
        ("Distributed strategy", strategy.strategy),
        ("Compile enabled", compile_cfg.get("enabled")),
        ("Compile backend", compile_cfg.get("backend")),
        ("Attention backend", attn.get("implementation")),
        ("Generation cache", gen.get("cache")),
    ]
    if primary:
        rows.extend(
            [
                ("Per-device batch", primary.per_device_batch_size),
                ("Gradient accumulation", primary.gradient_accumulation_steps),
                ("Global batch", primary.effective_global_batch_size),
            ]
        )
    rows.extend(
        [
            ("SID implementation", sid.get("implementation")),
            ("SID codebooks", sid.get("codebook_size")),
            ("Collision handling", sid.get("collision_handling")),
            ("Triton SID kernel", triton.get("rq_distance_argmin", False)),
        ]
    )
    width = max(len(k) for k, _ in rows)
    lines = ["════════ preflight ════════"]
    for k, v in rows:
        lines.append(f"{k:<{width}} : {v}")
    for note in info.get("sid_notes") or []:
        lines.append(f"ok: {note}")
    lines.append("═══════════════════════════")
    return "\n".join(lines)
