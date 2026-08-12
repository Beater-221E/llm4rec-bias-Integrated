#!/usr/bin/env python3
"""Resolve runtime strategies for simulated memory/world configurations.

Does not run expensive training. Prints plans for:
  1 GPU low-memory
  4 GPU low-memory
  4 GPU large-memory
  8 GPU large-memory

Uses free_vram_bytes + num_params (never GPU product names).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm4rec.runtime.hardware import HardwareInfo
from llm4rec.runtime.strategy import resolve_strategy


@dataclass
class Scenario:
    name: str
    world_size: int
    free_gb: float
    params_b: float
    has_ref: bool = False
    stage: str = "sft"


SCENARIOS = [
    Scenario("1gpu_low_mem", 1, 8.0, 0.5, False, "sft"),
    Scenario("4gpu_low_mem", 4, 8.0, 0.5, True, "grpo"),
    Scenario("4gpu_large_mem", 4, 40.0, 0.5, True, "grpo"),
    Scenario("8gpu_large_mem", 8, 40.0, 0.5, False, "sft"),
]


def _hw(world: int, free_gb: float) -> HardwareInfo:
    free = int(free_gb * (1024**3))
    return HardwareInfo(
        device_count=world,
        local_rank=0,
        world_size=world,
        gpu_name="sim",
        compute_capability=(8, 0),
        total_memory=free,
        free_memory=free,
        bf16_supported=True,
        tf32_supported=True,
        distributed=world > 1,
        cuda_available=True,
    )


def main() -> int:
    hw_cfg = {
        "strategy_auto": {
            "ddp_pressure_threshold": 0.45,
            "fsdp_pressure_threshold": 0.70,
        }
    }
    print("name\tworld\tfree_gb\tparams_b\tref\tstage\tstrategy\tsource\tpressure")
    for s in SCENARIOS:
        choice = resolve_strategy(
            "auto",
            _hw(s.world_size, s.free_gb),
            route="minionerec",
            mode="reproduction",
            model_params_b=s.params_b,
            precision="bf16",
            has_reference_model=s.has_ref,
            stage=s.stage,
            hw_cfg=hw_cfg,
            free_vram_bytes=int(s.free_gb * (1024**3)),
        )
        print(
            f"{s.name}\t{s.world_size}\t{s.free_gb}\t{s.params_b}\t{s.has_ref}\t"
            f"{s.stage}\t{choice.effective_strategy}\t{choice.source}\t"
            f"{None if choice.pressure_ratio is None else round(choice.pressure_ratio, 3)}"
        )
    # Sanity: low VRAM multi-GPU should prefer FSDP more often than large VRAM
    low = resolve_strategy(
        "auto",
        _hw(4, 8.0),
        model_params_b=0.5,
        has_reference_model=True,
        stage="grpo",
        precision="bf16",
        hw_cfg=hw_cfg,
        free_vram_bytes=int(8 * (1024**3)),
    )
    high = resolve_strategy(
        "auto",
        _hw(4, 40.0),
        model_params_b=0.5,
        has_reference_model=True,
        stage="grpo",
        precision="bf16",
        hw_cfg=hw_cfg,
        free_vram_bytes=int(40 * (1024**3)),
    )
    print(
        f"\n[check] low={low.effective_strategy} (p={low.pressure_ratio:.3f}) "
        f"high={high.effective_strategy} (p={high.pressure_ratio:.3f})"
    )
    if (low.pressure_ratio or 0) <= (high.pressure_ratio or 0):
        print("[warn] expected higher pressure on low-VRAM scenario")
        return 1
    if low.effective_strategy != "fsdp" or high.effective_strategy != "ddp":
        print(
            f"[warn] expected low→fsdp high→ddp, got {low.effective_strategy}/{high.effective_strategy}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
