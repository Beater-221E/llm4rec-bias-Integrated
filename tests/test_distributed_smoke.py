"""Optional 2-GPU smoke: verifies GRPO/DPO logging collectives do not deadlock.

Run manually:
  torchrun --standalone --nproc_per_node=2 -m pytest tests/test_distributed_smoke.py -q
Skipped automatically when WORLD_SIZE < 2.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist


def _distributed_ready() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) >= 2


@pytest.mark.skipif(not _distributed_ready(), reason="needs torchrun with >=2 processes")
def test_grpo_logging_collectives_no_deadlock():
    from llm4rec.core import distributed as dist_utils

    dist_utils.init_process_group()
    # Simulate several logging steps: every rank must enter all_reduce
    for step in range(1, 6):
        loss = float(dist_utils.rank() + step)
        reduced = dist_utils.all_reduce_mean(loss)
        if dist_utils.is_main():
            assert isinstance(reduced, float)
        dist_utils.barrier(f"step_{step}")
    dist_utils.cleanup()


@pytest.mark.skipif(not _distributed_ready(), reason="needs torchrun with >=2 processes")
def test_dpo_logging_collectives_no_deadlock():
    from llm4rec.core import distributed as dist_utils

    dist_utils.init_process_group()
    for step in range(1, 6):
        stats = {
            "loss": float(step),
            "dpo_margin": float(dist_utils.rank()),
            "accuracy": 0.5,
        }
        mean_loss = dist_utils.all_reduce_mean(stats["loss"])
        mean_margin = dist_utils.all_reduce_mean(stats["dpo_margin"])
        mean_acc = dist_utils.all_reduce_mean(stats["accuracy"])
        if dist_utils.is_main():
            assert mean_loss == pytest.approx(float(step))
            assert mean_acc == pytest.approx(0.5)
            assert isinstance(mean_margin, float)
        dist_utils.barrier(f"dpo_{step}")
    dist_utils.cleanup()
