"""Two-GPU FSDP policy/reference mixup sync validation.

Run via:
  torchrun --standalone --nproc_per_node=2 -m pytest tests/test_fsdp_reference_sync.py -q
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn


def _world() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2 or _world() < 2,
    reason="requires torchrun with >=2 GPUs",
)
def test_fsdp_reference_sync_two_gpu():
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from llm4rec.trainers.ref_sync import maybe_sync_reference_model

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.lin(x)

    torch.manual_seed(0)
    policy = Tiny().cuda(local)
    ref = Tiny().cuda(local)
    # Distinct starting points
    with torch.no_grad():
        policy.lin.weight.fill_(1.0)
        ref.lin.weight.fill_(0.0)

    policy = FSDP(policy)
    ref = FSDP(ref)
    for p in ref.parameters():
        p.requires_grad = False

    # Capture pre-sync reference shard
    ref_before = [p.detach().clone() for p in ref.parameters()]

    # One policy update
    opt = torch.optim.SGD(policy.parameters(), lr=0.1)
    x = torch.randn(4, 8, device=f"cuda:{local}")
    loss = policy(x).float().pow(2).mean()
    loss.backward()
    opt.step()

    synced = maybe_sync_reference_model(
        policy, ref, step=512, enabled=True, alpha=0.6, sync_steps=512
    )
    assert synced

    changed = False
    for before, after in zip(ref_before, ref.parameters()):
        if not torch.allclose(before, after.data):
            changed = True
            break
    assert changed, "reference shards should change after mixup sync"

    # Next forward must be finite
    y = ref(x)
    assert torch.isfinite(y).all()
