"""Hardware capability detection via PyTorch CUDA APIs (no GPU-name hardcoding)."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class HardwareInfo:
    device_count: int
    local_rank: int
    world_size: int
    gpu_name: str
    compute_capability: tuple[int, int] | None
    total_memory: int | None
    free_memory: int | None
    bf16_supported: bool
    tf32_supported: bool
    distributed: bool
    cuda_available: bool

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.compute_capability is not None:
            d["compute_capability"] = list(self.compute_capability)
        return d


def detect_hardware(*, local_rank: int | None = None, world_size: int | None = None) -> HardwareInfo:
    import torch

    lr = int(local_rank if local_rank is not None else os.environ.get("LOCAL_RANK", 0))
    ws = int(world_size if world_size is not None else os.environ.get("WORLD_SIZE", 1))
    cuda = torch.cuda.is_available()
    count = int(torch.cuda.device_count()) if cuda else 0

    gpu_name = "cpu"
    cc: tuple[int, int] | None = None
    total_mem = free_mem = None
    bf16 = False
    tf32 = False

    if cuda and count > 0:
        idx = min(lr, count - 1)
        props = torch.cuda.get_device_properties(idx)
        gpu_name = props.name
        cc = tuple(torch.cuda.get_device_capability(idx))  # type: ignore[assignment]
        total_mem = int(props.total_memory)
        try:
            free, total = torch.cuda.mem_get_info(idx)
            free_mem, total_mem = int(free), int(total)
        except Exception:  # noqa: BLE001
            free_mem = None
        bf16 = bool(torch.cuda.is_bf16_supported())
        # TF32 is an Ampere+ matmul mode (cc >= 8.0)
        tf32 = bool(cc is not None and cc[0] >= 8)

    return HardwareInfo(
        device_count=count,
        local_rank=lr,
        world_size=ws,
        gpu_name=gpu_name,
        compute_capability=cc,
        total_memory=total_mem,
        free_memory=free_mem,
        bf16_supported=bf16,
        tf32_supported=tf32,
        distributed=ws > 1,
        cuda_available=cuda,
    )


def apply_nccl_compat_profile() -> dict[str, str]:
    """Optionally disable P2P/IB when ``LLM4REC_NCCL_COMPAT=1``.

    Normal path: leave NCCL topology detection alone.
    """
    applied: dict[str, str] = {}
    if os.environ.get("LLM4REC_NCCL_COMPAT", "0") not in {"1", "true", "TRUE", "yes"}:
        return applied
    defaults = {
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "NCCL_NVML_ENABLE": "0",
    }
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def configure_tf32(enabled: bool) -> None:
    import torch

    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
