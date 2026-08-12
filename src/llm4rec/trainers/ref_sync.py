"""TR-DPO / MiniOneRec reference-model mixup sync.

Official MiniOneRec uses TRL ``SyncRefModelCallback``:
every ``ref_model_sync_steps`` (default 512),
``π_ref ← α·π_θ + (1−α)·π_ref`` with ``α=0.6``.

Under FSDP, parameter views are local/sharded shards. Interpolating
corresponding local shards with the same α on every rank is valid because
FSDP keeps identical shard layouts for identically wrapped modules.
We never zip flattened FSDP state with unwrapped modules.
"""

from __future__ import annotations

from typing import Any

import torch

from llm4rec.core.distributed import unwrap


def _is_fsdp(module: Any) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return isinstance(module, FSDP)
    except Exception:  # noqa: BLE001
        return False


@torch.no_grad()
def _mix_param_pairs(
    policy_params,
    ref_params,
    *,
    alpha: float,
) -> int:
    """Local lerp on matching shapes. Returns number of updated tensors."""
    n = 0
    for p_param, r_param in zip(policy_params, ref_params, strict=False):
        if r_param.data.shape != p_param.data.shape:
            continue
        # In-place: r ← (1-α)·r + α·p  (identical on every rank for FSDP shards)
        r_param.data.mul_(1.0 - alpha).add_(p_param.data, alpha=alpha)
        n += 1
    return n


@torch.no_grad()
def maybe_sync_reference_model(
    policy: Any,
    ref: Any,
    step: int,
    *,
    enabled: bool = False,
    alpha: float = 0.6,
    sync_steps: int = 512,
) -> bool:
    """Mix policy into reference when ``enabled`` and ``step % sync_steps == 0``.

    Returns True if a sync was performed.
    """
    if not enabled or ref is None or policy is None:
        return False
    if sync_steps <= 0 or step <= 0 or step % int(sync_steps) != 0:
        return False
    alpha_f = min(max(float(alpha), 0.0), 1.0)

    # FSDP-aware path: operate on the wrapped modules' local parameters so
    # shard layouts stay aligned. Do not unwrap before iterating params.
    if _is_fsdp(policy) and _is_fsdp(ref):
        _mix_param_pairs(policy.parameters(), ref.parameters(), alpha=alpha_f)
        return True
    if _is_fsdp(policy) ^ _is_fsdp(ref):
        raise RuntimeError(
            "FSDP reference sync requires both policy and reference to be FSDP-wrapped "
            "(or both unwrapped). Mixed wrapping is unsafe for parameter-wise mixup."
        )

    pol = unwrap(policy)
    reference = unwrap(ref)
    _mix_param_pairs(pol.parameters(), reference.parameters(), alpha=alpha_f)
    return True
