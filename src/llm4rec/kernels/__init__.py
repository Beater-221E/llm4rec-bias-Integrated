"""Optional custom kernels. Triton RQ distance is opt-in and off in reproduction."""

from __future__ import annotations

from typing import Literal

import torch

Backend = Literal["auto", "reference", "triton"]


def quantize_nearest(
    x: torch.Tensor,
    codebook: torch.Tensor,
    *,
    backend: Backend = "auto",
) -> torch.Tensor:
    """Nearest-code indices. Custom Triton is disabled unless explicitly requested."""
    if backend == "triton":
        try:
            from llm4rec.kernels.rq_distance import triton_argmin_distance

            return triton_argmin_distance(x, codebook)
        except Exception:
            pass  # fall through to reference
    # reference: materialize distances (correct, used in reproduction)
    distances = (
        torch.sum(x**2, dim=1, keepdim=True)
        + torch.sum(codebook**2, dim=1, keepdim=True).t()
        - 2 * torch.matmul(x, codebook.t())
    )
    return torch.argmin(distances, dim=-1)
