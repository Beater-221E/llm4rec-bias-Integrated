"""SID collision resolution and metrics wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from llm4rec.sid.base import (
    CollisionMetrics,
    collision_groups,
    collision_rate,
    compute_collision_metrics,
    find_duplicate_text_groups,
)

# Re-export for callers
__all__ = [
    "CollisionMetrics",
    "collision_groups",
    "collision_rate",
    "compute_collision_metrics",
    "find_duplicate_text_groups",
    "resolve_collisions_sinkhorn",
    "enforce_unique_last_code",
]


def enforce_unique_last_code(codes: np.ndarray, codebook_size: int) -> np.ndarray:
    """Experimental: reshuffle last digit within shared prefixes (rqkmeans path)."""
    from collections import defaultdict

    out = codes.copy()
    last = codes.shape[1] - 1
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(codes):
        groups[tuple(int(c) for c in row[:last])].append(i)
    for prefix, members in groups.items():
        if len(members) <= 1:
            continue
        members.sort(key=lambda i: (int(codes[i, last]), i))
        if len(members) > codebook_size:
            for j, i in enumerate(members):
                out[i, last] = j % codebook_size
                out[i, last - 1] = (prefix[last - 1] + j // codebook_size) % codebook_size
        else:
            for j, i in enumerate(members):
                out[i, last] = j
    return out


def resolve_collisions_sinkhorn(
    model: Any,
    x: torch.Tensor,
    codes: np.ndarray,
    *,
    sk_epsilon: float = 0.003,
    max_iters: int = 20,
    device: str = "cuda:0",
    log: Any = print,
) -> np.ndarray:
    """Integrated/simple RQVAE collision post-process (last-level Sinkhorn).

    For the official MiniOneRec model, prefer
    ``llm4rec.sid.minionerec_rqvae.resolve_collisions_minionerec``.
    """
    # Dispatch to official path when possible
    if hasattr(model, "rq") and hasattr(model, "get_indices"):
        from llm4rec.sid.minionerec_rqvae import resolve_collisions_minionerec

        return resolve_collisions_minionerec(
            model,
            x,
            codes,
            sk_epsilon=sk_epsilon,
            max_iters=max_iters,
            device=device,
            log=log,
        )

    # Fallback: simplified RQVAE API (encode_indices + sk_epsilons list)
    from llm4rec.sid.rqvae import RQVAE

    if not isinstance(model, RQVAE):
        raise TypeError(f"unsupported SID model type for Sinkhorn resolution: {type(model)}")

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    model.sk_epsilons = [0.0] * (model.num_layers - 1) + [sk_epsilon]

    all_x = x.to(dev)
    out = codes.copy()
    it = 0
    while True:
        unique = len({tuple(int(c) for c in row) for row in out})
        rate = (len(out) - unique) / len(out)
        if it >= max_iters or unique == len(out):
            log(f"[sid] Sinkhorn post-process: iters={it} collision={rate:.4f}")
            break
        groups = collision_groups(out)
        log(f"[sid]   round {it + 1}: {len(groups)} collision groups")
        for members in groups:
            batch = all_x[torch.tensor(members, device=dev)]
            idx = model.encode_indices(batch, use_sk=True).cpu().numpy()
            for pos, item_idx in enumerate(members):
                out[item_idx] = idx[pos]
        it += 1
    return out
