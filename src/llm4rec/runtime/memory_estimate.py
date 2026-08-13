"""Static training memory estimates for strategy:auto (no activation analytics)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


_BYTES = {
    "fp32": 4,
    "float32": 4,
    "fp16": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
}


@dataclass
class MemoryEstimate:
    num_params: int
    bytes_per_param: int
    policy_static_bytes: int
    reference_static_bytes: int
    optimizer_static_bytes: int
    estimated_static_training_bytes: int
    free_vram_bytes: int | None
    pressure_ratio: float | None
    has_reference_model: bool

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Friendly GB fields for manifests
        d["policy_static_gb"] = round(self.policy_static_bytes / (1024**3), 3)
        d["reference_static_gb"] = round(self.reference_static_bytes / (1024**3), 3)
        d["optimizer_static_gb"] = round(self.optimizer_static_bytes / (1024**3), 3)
        d["estimated_static_training_gb"] = round(
            self.estimated_static_training_bytes / (1024**3), 3
        )
        if self.free_vram_bytes is not None:
            d["free_vram_gb"] = round(self.free_vram_bytes / (1024**3), 3)
        return d


def bytes_per_param(precision: str) -> int:
    return int(_BYTES.get(str(precision).lower(), 4))


def estimate_static_training_bytes(
    num_params: int,
    *,
    precision: str = "bf16",
    optimizer: str = "adamw",
    has_reference_model: bool = False,
    free_vram_bytes: int | None = None,
) -> MemoryEstimate:
    """Conservative static footprint: params + grads + optimizer (+ optional ref).

    Does **not** model activations; used only as a relative pressure signal for
    strategy:auto / activation-checkpointing:auto.
    """
    bpp = bytes_per_param(precision)
    # Master weights always materialize in the training dtype (approx).
    policy = int(num_params) * bpp
    # Gradients match training dtype.
    grads = int(num_params) * bpp
    opt_name = str(optimizer or "adamw").lower()
    if "sgd" in opt_name and "adam" not in opt_name:
        # Momentum buffer
        optim = int(num_params) * bpp
    else:
        # AdamW: m + v in fp32 typically
        optim = int(num_params) * 8
    ref = int(num_params) * bpp if has_reference_model else 0
    total = policy + grads + optim + ref
    pressure = None
    if free_vram_bytes is not None and free_vram_bytes > 0:
        pressure = float(total) / float(free_vram_bytes)
    return MemoryEstimate(
        num_params=int(num_params),
        bytes_per_param=bpp,
        policy_static_bytes=policy,
        reference_static_bytes=ref,
        optimizer_static_bytes=optim,
        estimated_static_training_bytes=total,
        free_vram_bytes=free_vram_bytes,
        pressure_ratio=pressure,
        has_reference_model=bool(has_reference_model),
    )


def select_memory_probe_examples(
    dataset: Any,
    *,
    max_scan: int = 128,
    percentile: float = 0.90,
    batch_size: int = 1,
    seed: int | None = 42,
    length_fn=None,
) -> list[int]:
    """Pick indices around a length percentile for representative memory probes."""
    n = len(dataset)
    if n <= 0:
        return []
    scan = min(int(max_scan), n)
    if seed is not None:
        import random

        rng = random.Random(int(seed))
        candidates = list(range(n))
        rng.shuffle(candidates)
        candidates = candidates[:scan]
    else:
        candidates = list(range(scan))

    def _len(i: int) -> int:
        if length_fn is not None:
            return int(length_fn(dataset[i]))
        row = dataset[i]
        if isinstance(row, dict) and "input_ids" in row:
            return len(row["input_ids"])
        return 0

    scored = [(i, _len(i)) for i in candidates]
    scored.sort(key=lambda x: x[1])
    if not scored:
        return [0] * max(1, int(batch_size))
    pct = min(max(float(percentile), 0.0), 1.0)
    target_idx = min(len(scored) - 1, int(pct * (len(scored) - 1)))
    # Take a contiguous window around the percentile for the microbatch
    bs = max(1, int(batch_size))
    start = max(0, target_idx - bs // 2)
    end = min(len(scored), start + bs)
    start = max(0, end - bs)
    return [scored[j][0] for j in range(start, end)]
