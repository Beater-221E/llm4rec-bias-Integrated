"""Distributed strategy auto-detection (single GPU vs multi-GPU)."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from llm4rec_bias_Integrated.core.exceptions import ConfigurationError
from llm4rec_bias_Integrated.models.base import require_cuda

logger = logging.getLogger("llm4rec_bias_Integrated")

DistributedBackend = Literal["single", "ddp", "accelerate", "deepspeed_zero2", "deepspeed_zero3"]


@dataclass(frozen=True)
class DistributedPlan:
    """Resolved launch plan for the current process / hardware."""

    strategy: DistributedBackend
    world_size: int
    local_rank: int
    global_rank: int
    is_main_process: bool
    nproc_per_node: int
    effective_batch_size: int
    launch_hint: str
    details: dict[str, Any]


def _env_int(name: str, default: int = -1) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def nvml_usable() -> bool:
    """Return False when NVML cannot init (common NCCL multi-GPU blocker)."""
    try:
        count = torch.cuda._raw_device_count_nvml()  # type: ignore[attr-defined]
        return count is not None and int(count) > 0
    except Exception:
        return False


def detect_distributed_context() -> dict[str, Any]:
    """Read torchrun / Accelerate env vars if present."""
    require_cuda()
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)
    rank = _env_int("RANK", 0)
    already_launched = world_size > 1 or "LOCAL_RANK" in os.environ
    return {
        "world_size": max(world_size, 1),
        "local_rank": local_rank,
        "rank": rank,
        "already_launched": already_launched,
        "visible_gpus": int(torch.cuda.device_count()),
        "nvml_usable": nvml_usable(),
    }


def resolve_distributed_plan(
    training_cfg: dict[str, Any],
    *,
    model_name: str | None = None,
) -> DistributedPlan:
    """Choose a distributed strategy from config + hardware.

    Auto rules (GPU only — never CPU):
      - 1 visible GPU → single
      - ≥2 GPUs + healthy NVML → accelerate
      - ≥2 GPUs + broken NVML → single CUDA with warning
      - Explicit ddp/accelerate with broken NVML → ConfigurationError
    """
    require_cuda()
    ctx = detect_distributed_context()
    requested = str(training_cfg.get("distributed") or "auto").lower()
    batch = int(training_cfg.get("batch_size", 1))
    accum = int(training_cfg.get("gradient_accumulation_steps", 1))
    n_gpus = ctx["visible_gpus"]
    if n_gpus < 1:
        raise ConfigurationError("No CUDA devices visible for distributed planning.")

    notes: list[str] = []
    if requested == "auto":
        if ctx["already_launched"] and ctx["world_size"] > 1:
            strategy: DistributedBackend = "accelerate"
        elif n_gpus == 1:
            strategy = "single"
        elif not ctx["nvml_usable"]:
            strategy = "single"
            notes.append(
                f"{n_gpus} GPUs visible but NVML is unusable "
                "(driver/library mismatch). Auto strategy uses single-GPU CUDA. "
                "Fix the NVIDIA driver stack before enabling multi-GPU NCCL."
            )
            logger.warning(notes[-1])
        else:
            strategy = "accelerate"
    elif requested in {"single", "ddp", "accelerate", "deepspeed_zero2", "deepspeed_zero3"}:
        strategy = requested  # type: ignore[assignment]
        if (
            strategy != "single"
            and n_gpus > 1
            and not ctx["nvml_usable"]
            and not ctx["already_launched"]
        ):
            raise ConfigurationError(
                f"training.distributed={strategy} requires working NVML/NCCL, "
                "but nvmlInit failed (driver/library version mismatch). "
                "Use training.distributed=single/auto until the host driver is fixed."
            )
    else:
        raise ConfigurationError(f"Unknown training.distributed={requested!r}")

    if strategy == "single":
        world = 1
        local_rank = 0
        rank = 0
        nproc = 1
        launch_hint = "python -m llm4rec_bias_Integrated.cli.main train ..."
    elif ctx["already_launched"]:
        world = ctx["world_size"]
        local_rank = ctx["local_rank"]
        rank = ctx["rank"]
        nproc = world
        launch_hint = "(already under torchrun/accelerate)"
    else:
        world = n_gpus
        local_rank = 0
        rank = 0
        nproc = n_gpus
        if strategy.startswith("deepspeed"):
            launch_hint = (
                f"python -m accelerate.commands.launch --num_processes {nproc} "
                f"--use_deepspeed -m llm4rec_bias_Integrated.cli.main train ..."
            )
        else:
            launch_hint = (
                f"python -m accelerate.commands.launch --num_processes {nproc} "
                f"--multi_gpu -m llm4rec_bias_Integrated.cli.main train ..."
            )

    effective = batch * accum * max(world, 1)
    details = {
        "requested": requested,
        "visible_gpus": n_gpus,
        "already_launched": ctx["already_launched"],
        "nvml_usable": ctx["nvml_usable"],
        "model_name": model_name,
        "per_device_batch_size": batch,
        "gradient_accumulation_steps": accum,
        "notes": notes,
    }
    return DistributedPlan(
        strategy=strategy,
        world_size=world,
        local_rank=local_rank,
        global_rank=rank,
        is_main_process=(rank == 0),
        nproc_per_node=nproc,
        effective_batch_size=effective,
        launch_hint=launch_hint,
        details=details,
    )


def maybe_init_process_group(plan: DistributedPlan) -> None:
    """Set the current CUDA device for this process."""
    require_cuda()
    if plan.local_rank >= 0:
        torch.cuda.set_device(plan.local_rank % max(torch.cuda.device_count(), 1))


def _rendezvous_dir() -> Path:
    # Prefer MASTER_ADDR/PORT only — TORCHELASTIC_RUN_ID can be missing/inconsistent
    # across ranks, which splits file barriers and deadlocks the job.
    job = (
        os.environ.get("LLM4REC_FULL_JOB_ID")
        or f"{os.environ.get('MASTER_ADDR', '127.0.0.1')}_"
        f"{os.environ.get('MASTER_PORT', '29500')}_"
        f"{os.environ.get('WORLD_SIZE', '1')}"
    )
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in job)
    return Path(tempfile.gettempdir()) / f"llm4rec_bias_Integrated_rendezvous_{safe}"


def allocate_shared_run_dir(
    config: dict[str, Any],
    plan: DistributedPlan,
    *,
    build_run_dir,
    timeout_s: float = 120.0,
) -> Path:
    """Ensure all ranks under torchrun/accelerate share one run directory.

    Without this, each rank calls ``build_run_dir`` independently and gets a
    different ``<ts>_<id>/``, so non-zero ranks miss the SFT adapter at GRPO time.
    """
    if plan.world_size <= 1 or not plan.details.get("already_launched"):
        return build_run_dir(config)

    sync = _rendezvous_dir()
    marker = sync / "run_dir.txt"
    ready = sync / "run_dir.ready"
    token_path = sync / "run_dir.token"

    if plan.is_main_process:
        out = build_run_dir(config)
        sync.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        # Drop stale marks from a previous crashed job on the same MASTER_PORT.
        for stale in (marker, ready, token_path):
            if stale.exists():
                stale.unlink(missing_ok=True)
        marker.write_text(str(out.resolve()), encoding="utf-8")
        token_path.write_text(token, encoding="utf-8")
        ready.write_text(token, encoding="utf-8")
        logger.info("Shared run directory (rank0): %s", out)
        return out

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if ready.is_file() and marker.is_file() and token_path.is_file():
            token = token_path.read_text(encoding="utf-8").strip()
            ready_tok = ready.read_text(encoding="utf-8").strip()
            text = marker.read_text(encoding="utf-8").strip()
            if token and token == ready_tok and text:
                out = Path(text)
                logger.info("Shared run directory (rank%s): %s", plan.global_rank, out)
                return out
        time.sleep(0.05)
    raise ConfigurationError(
        f"Rank {plan.global_rank} timed out waiting for shared run_dir "
        f"({timeout_s:.0f}s, marker={marker})"
    )


def distributed_barrier(
    plan: DistributedPlan,
    *,
    name: str = "default",
    prefer_file: bool = False,
    timeout_s: float = 1800.0,
) -> None:
    """Synchronize ranks after main-process-only artifacts (e.g. SFT adapter save).

    Use ``prefer_file=True`` when rank0 will do long GPU-only work (held-out eval)
    while others wait — an NCCL barrier would spin GPUs at ~100% util / low power.
    """
    if plan.world_size <= 1:
        return
    if (
        not prefer_file
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.barrier()
        return
    sync = _rendezvous_dir()
    gate = sync / f"barrier_{name}_{plan.world_size}"
    open_mark = gate / "_open"
    if plan.is_main_process:
        token = uuid.uuid4().hex
        if gate.exists():
            for p in gate.iterdir():
                p.unlink(missing_ok=True)
        else:
            gate.mkdir(parents=True, exist_ok=True)
        open_mark.write_text(token, encoding="utf-8")
    else:
        deadline_open = time.time() + min(timeout_s, 120.0)
        token = ""
        while time.time() < deadline_open:
            if open_mark.is_file():
                token = open_mark.read_text(encoding="utf-8").strip()
                if token:
                    break
            time.sleep(0.05)
        if not token:
            raise ConfigurationError(
                f"Rank {plan.global_rank} timed out waiting for barrier open "
                f"(name={name}, gate={gate})"
            )
    if plan.is_main_process:
        token = open_mark.read_text(encoding="utf-8").strip()
    (gate / f"rank_{plan.global_rank}.done").write_text(token, encoding="utf-8")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        done = list(gate.glob("rank_*.done"))
        if len(done) >= plan.world_size:
            # Require matching generation tokens so stale marks from a prior
            # job on the same MASTER_PORT cannot release the barrier early.
            toks = {p.read_text(encoding="utf-8").strip() for p in done}
            if len(toks) == 1 and token in toks:
                return
        time.sleep(0.05)
    raise ConfigurationError(
        f"Rank {plan.global_rank} timed out in distributed_barrier "
        f"(name={name}, world={plan.world_size}, gate={gate})"
    )


def wait_for_file(path: Path | str, *, timeout_s: float = 180.0) -> Path:
    """Poll until ``path`` exists (used before PeftModel.from_pretrained on all ranks)."""
    target = Path(path)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if target.is_file():
            return target
        time.sleep(0.05)
    raise ConfigurationError(f"Timed out waiting for artifact: {target}")
