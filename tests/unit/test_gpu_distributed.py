"""Unit tests for GPU-only precision and distributed planning."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from llm4rec_bias_Integrated.core.exceptions import ConfigurationError
from llm4rec_bias_Integrated.models.base import require_cuda, resolve_precision
from llm4rec_bias_Integrated.prompts.candidate_choice import parse_choice
from llm4rec_bias_Integrated.trainers.distributed import (
    DistributedPlan,
    allocate_shared_run_dir,
    resolve_distributed_plan,
    wait_for_file,
)


def test_require_cuda_or_skip() -> None:
    if not torch.cuda.is_available():
        with pytest.raises(ConfigurationError):
            require_cuda()
    else:
        require_cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_auto_precision_fp16_on_pre_ampere() -> None:
    major, _ = torch.cuda.get_device_capability()
    prec = resolve_precision("auto")
    if major < 8:
        assert prec.fp16 is True
        assert prec.bf16 is False
    else:
        assert prec.bf16 is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_distributed_auto_plan() -> None:
    from llm4rec_bias_Integrated.trainers.distributed import nvml_usable

    n = torch.cuda.device_count()
    plan = resolve_distributed_plan(
        {"distributed": "auto", "batch_size": 2, "gradient_accumulation_steps": 4}
    )
    if n == 1:
        assert plan.strategy == "single"
        assert plan.effective_batch_size == 8
    elif not nvml_usable():
        # Host cannot NCCL — auto must stay on single CUDA device (not CPU).
        assert plan.strategy == "single"
        assert plan.world_size == 1
        assert plan.details["nvml_usable"] is False
    else:
        assert plan.strategy == "accelerate"
        assert plan.nproc_per_node == n
        assert plan.effective_batch_size == 2 * 4 * n


def test_parse_choice_shared() -> None:
    assert parse_choice("A", 10) == 0
    assert parse_choice("Based on", 10) is None


def _plan(*, rank: int, world: int, launched: bool) -> DistributedPlan:
    return DistributedPlan(
        strategy="accelerate" if world > 1 else "single",
        world_size=world,
        local_rank=rank,
        global_rank=rank,
        is_main_process=(rank == 0),
        nproc_per_node=world,
        effective_batch_size=8,
        launch_hint="test",
        details={"already_launched": launched, "visible_gpus": world},
    )


def test_allocate_shared_run_dir_single_calls_builder(tmp_path: Path) -> None:
    created: list[Path] = []

    def _build(_cfg: dict) -> Path:
        p = tmp_path / "run_a"
        p.mkdir()
        created.append(p)
        return p

    out = allocate_shared_run_dir(
        {}, _plan(rank=0, world=1, launched=False), build_run_dir=_build
    )
    assert out == tmp_path / "run_a"
    assert created == [out]


def test_allocate_shared_run_dir_multigpu_rendezvous(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLM4REC_FULL_JOB_ID", f"unit_{tmp_path.name}")
    sync_root = tmp_path / "rendezvous"
    monkeypatch.setattr(
        "llm4rec_bias_Integrated.trainers.distributed._rendezvous_dir",
        lambda: sync_root,
    )

    def _build(_cfg: dict) -> Path:
        p = tmp_path / "shared_run"
        p.mkdir()
        return p

    main = allocate_shared_run_dir(
        {}, _plan(rank=0, world=4, launched=True), build_run_dir=_build
    )
    worker = allocate_shared_run_dir(
        {},
        _plan(rank=1, world=4, launched=True),
        build_run_dir=lambda _c: (_ for _ in ()).throw(AssertionError("worker must not build")),
    )
    assert main == worker == tmp_path / "shared_run"


def test_wait_for_file(tmp_path: Path) -> None:
    target = tmp_path / "adapter_config.json"
    target.write_text("{}", encoding="utf-8")
    assert wait_for_file(target, timeout_s=1.0) == target
