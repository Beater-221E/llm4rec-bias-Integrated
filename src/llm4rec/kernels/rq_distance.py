"""Optional fused RQ distance+argmin.

Status: **evaluated, not justified** for the current SID encode path.
The function still delegates to reference matmul distance+argmin so that
``backend=triton`` never silently changes indices. Reproduction keeps
``optimization.triton.rq_distance_argmin: false`` / ``backend=reference``.

A real Triton fused kernel may be added only after (1) encode profiling shows
this op is hot and (2) exact index parity tests pass.
"""

from __future__ import annotations

import torch


def triton_argmin_distance(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Placeholder: identical math to the reference path (not a fused kernel)."""
    try:
        import triton  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "triton is not installed; use backend='reference' or install triton"
        ) from exc
    distances = (
        torch.sum(x**2, dim=1, keepdim=True)
        + torch.sum(codebook**2, dim=1, keepdim=True).t()
        - 2 * torch.matmul(x, codebook.t())
    )
    return torch.argmin(distances, dim=-1)
