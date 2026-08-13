"""Packed distributed metric reductions for GRPO / DPO logging."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.distributed as dist

from llm4rec.core import distributed as dist_utils


def reduce_scalar_pack(values: Sequence[float], *, op: str = "mean") -> list[float]:
    """All-reduce a small pack of scalars in ONE collective.

    ``op='mean'`` divides by world size; ``op='sum'`` does not.
    """
    if not values:
        return []
    if not dist_utils.is_distributed():
        return [float(v) for v in values]
    device = (
        f"cuda:{dist_utils.local_rank()}" if torch.cuda.is_available() else "cpu"
    )
    tensor = torch.tensor([float(v) for v in values], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if op == "mean":
        tensor = tensor / dist_utils.world_size()
    return [float(x) for x in tensor.tolist()]


def reduce_reward_stats(rewards: Sequence[float]) -> dict[str, float]:
    """Globally correct reward mean/std/max/min via sum / sumsq / count / max / min."""
    n = len(rewards)
    if n == 0:
        return {
            "reward": 0.0,
            "reward_std": 0.0,
            "reward_max": 0.0,
            "reward_min": 0.0,
            "reward_count": 0.0,
        }
    s = float(sum(rewards))
    s2 = float(sum(r * r for r in rewards))
    rmax = float(max(rewards))
    rmin = float(min(rewards))
    count = float(n)

    if not dist_utils.is_distributed():
        mean = s / count
        var = max(0.0, s2 / count - mean * mean)
        return {
            "reward": mean,
            "reward_std": var**0.5,
            "reward_max": rmax,
            "reward_min": rmin,
            "reward_count": count,
        }

    device = (
        f"cuda:{dist_utils.local_rank()}" if torch.cuda.is_available() else "cpu"
    )
    # pack: sum, sumsq, count, max, min  — max/min use MAX/MIN via trick:
    # we all_reduce SUM for sum/sumsq/count and separate max/min
    pack = torch.tensor([s, s2, count], device=device, dtype=torch.float64)
    dist.all_reduce(pack, op=dist.ReduceOp.SUM)
    tmax = torch.tensor([rmax], device=device, dtype=torch.float64)
    tmin = torch.tensor([rmin], device=device, dtype=torch.float64)
    dist.all_reduce(tmax, op=dist.ReduceOp.MAX)
    dist.all_reduce(tmin, op=dist.ReduceOp.MIN)
    gs, gs2, gcount = (float(x) for x in pack.tolist())
    mean = gs / max(gcount, 1.0)
    var = max(0.0, gs2 / max(gcount, 1.0) - mean * mean)
    return {
        "reward": mean,
        "reward_std": var**0.5,
        "reward_max": float(tmax.item()),
        "reward_min": float(tmin.item()),
        "reward_count": gcount,
    }
