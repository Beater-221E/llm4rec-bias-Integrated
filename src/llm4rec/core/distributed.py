"""分布式工具 —— 手写训练循环（SFT / GRPO / DPO）的多卡支持。

默认并行是 MiniOneRec 的分片：各 rank 切不同样本，本地 backward 后
``allreduce_gradients`` 平均梯度。不包 DDP，避免 V100 16GB 上 DDP bucket
把 logits / Adam 挤爆。

DeepSpeed ZeRO 仍走 HF Trainer；FSDP 仅在 ``strategy=fsdp`` 时 wrap。

这里提供：
    * rank / world_size / is_main 查询
    * ``wrap_ddp`` / ``wrap_fsdp``  可选包装（SFT/GRPO/DPO 默认不用 DDP）
    * ``shard``         按 rank 切分训练样本，各卡跑不同数据
    * ``allreduce_gradients``  无 DDP 时的梯度平均
    * ``all_reduce_mean`` 汇总标量指标（loss / reward / kl）
    * ``barrier``       阶段同步
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, Sequence, TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")


# ------------------------------------------------------------------ 查询


def rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main() -> bool:
    return rank() == 0


def is_distributed() -> bool:
    return world_size() > 1


# ------------------------------------------------------------------ 初始化


def init_process_group() -> bool:
    """torchrun 环境下初始化进程组。返回是否真的是多卡。"""
    if not is_distributed():
        return False
    if dist.is_initialized():
        return True
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    # Bind CUDA device *before* NCCL init; pass device_id so barriers/collectives
    # do not guess the wrong GPU (PyTorch warning → hang on V100 multi-GPU).
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
        dist.init_process_group(
            backend=backend,
            device_id=torch.device(f"cuda:{local_rank()}"),
        )
    else:
        dist.init_process_group(backend=backend)
    return True


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(name: str = "") -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[local_rank()])
    else:
        dist.barrier()


# ------------------------------------------------------------------ 模型


def wrap_ddp(model: Any, *, find_unused_parameters: bool = False) -> Any:
    """包成 DDP，让 ``loss.backward()`` 自动 all-reduce 梯度。

    Default ``find_unused_parameters=False``: zero advantages still execute the
    full transformer graph, so parameters are not unused. Override only for
    routes that genuinely skip subgraphs.
    """
    if not is_distributed():
        return model
    from torch.nn.parallel import DistributedDataParallel

    device_ids = [local_rank()] if torch.cuda.is_available() else None
    return DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=local_rank() if torch.cuda.is_available() else None,
        find_unused_parameters=bool(find_unused_parameters),
    )


def wrap_fsdp(
    model: Any,
    *,
    param_dtype: torch.dtype | None = None,
    reduce_dtype: torch.dtype | None = None,
    buffer_dtype: torch.dtype | None = None,
) -> Any:
    """Optional FSDP wrap. Precision comes from RuntimeContext, not GPU capability."""
    if not is_distributed():
        return model
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision
    except ImportError:
        return wrap_ddp(model, find_unused_parameters=False)

    mp = None
    # Only enable MixedPrecision when an explicit non-fp32 dtype is provided.
    if param_dtype is not None and param_dtype != torch.float32:
        mp = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype or param_dtype,
            buffer_dtype=buffer_dtype or param_dtype,
        )
    return FSDP(
        model,
        mixed_precision=mp,
        device_id=local_rank() if torch.cuda.is_available() else None,
    )


def is_fsdp(model: Any) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return isinstance(model, FSDP)
    except ImportError:
        return False


def unwrap(model: Any) -> Any:
    """取回底层模型（存盘、generate 都要用没包 DDP 的那个）。"""
    return getattr(model, "module", model)


def save_pretrained_distributed(
    model: Any,
    output_dir: Any,
    *,
    tokenizer: Any = None,
    is_main: bool | None = None,
) -> None:
    """Save HF weights: FSDP full-state gather on rank0; DDP unwrap otherwise."""
    from pathlib import Path

    out = Path(output_dir)
    main = is_main if is_main is not None else globals()["is_main"]()
    if is_fsdp(model):
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        except ImportError:
            if main:
                out.mkdir(parents=True, exist_ok=True)
                unwrap(model).save_pretrained(str(out))
                if tokenizer is not None:
                    tokenizer.save_pretrained(str(out))
            return
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state = model.state_dict()
        if main:
            out.mkdir(parents=True, exist_ok=True)
            # Prefer HF save via unwrapped module if available
            inner = getattr(model, "module", model)
            if hasattr(inner, "save_pretrained"):
                # Write gathered tensors onto a CPU copy of config-bearing module
                inner.save_pretrained(str(out), state_dict=state)
            else:
                torch.save(state, out / "pytorch_model.bin")
            if tokenizer is not None:
                tokenizer.save_pretrained(str(out))
        return

    if main:
        out.mkdir(parents=True, exist_ok=True)
        unwrap(model).save_pretrained(str(out))
        if tokenizer is not None:
            tokenizer.save_pretrained(str(out))


@contextmanager
def no_sync(model: Any) -> Iterator[None]:
    """梯度累积中间步跳过 all-reduce —— 只在最后一个 micro-step 同步。

    不这么做的话，accum=8 就会白白多做 7 次全量梯度通信。
    """
    if is_distributed() and hasattr(model, "no_sync"):
        with model.no_sync():
            yield
    else:
        yield


# ------------------------------------------------------------------ 数据


def shard(items: Sequence[T]) -> list[T]:
    """按 rank 切分。各卡跑不同样本，合起来才是一个完整 epoch。"""
    if not is_distributed():
        return list(items)
    r, w = rank(), world_size()
    return [item for i, item in enumerate(items) if i % w == r]


def allreduce_gradients(model: Any) -> None:
    """Average grads without a second DDP forward (that path hung on V100).

    Every trainable parameter participates, even if this rank has no grad
    (zero-filled). Skipping ``None`` grads made ranks allreduce different
    tensor counts and hung NCCL.
    """
    if not is_distributed():
        return
    core = unwrap(model)
    world = world_size()
    for p in core.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world)


def all_reduce_mean(value: float) -> float:
    """跨卡求均值 —— 日志里的 loss/reward 应该是全局的，不是本卡的。"""
    if not is_distributed():
        return float(value)
    tensor = torch.tensor(
        [float(value)],
        device=f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu",
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / world_size())


def all_reduce_min_int(value: int) -> int:
    """跨卡取最小值 —— memory-auto 探测后统一 micro-batch，避免 DDP 步数错位死锁。"""
    if not is_distributed():
        return int(value)
    device = f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu"
    tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return int(tensor.item())


def all_gather_object(obj: Any) -> list[Any]:
    if not is_distributed():
        return [obj]
    buckets: list[Any] = [None] * world_size()
    dist.all_gather_object(buckets, obj)
    return buckets


def summary_line() -> str:
    if not is_distributed():
        return "单卡"
    return (
        f"多卡分片：rank {rank()}/{world_size()} (local_rank={local_rank()})"
    )


def print_distributed_banner(log=print) -> None:
    """Rank-0 only summary of distributed / NCCL settings."""
    if not is_main():
        return
    import os

    log(f"[dist] {summary_line()}")
    if is_distributed():
        log(
            f"[dist] NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', '<unset>')} "
            f"NCCL_IB_DISABLE={os.environ.get('NCCL_IB_DISABLE', '<unset>')} "
            f"LLM4REC_NCCL_COMPAT={os.environ.get('LLM4REC_NCCL_COMPAT', '0')}"
        )
