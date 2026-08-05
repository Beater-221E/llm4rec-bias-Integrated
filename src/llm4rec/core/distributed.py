"""分布式工具 —— 手写训练循环（GRPO / DPO）的多卡支持。

背景：SFT 走 HF ``Trainer``，torchrun 下它自己会包 DDP、同步梯度、只让 rank0
存盘。但 GRPO / DPO 是我们手写的循环，**必须自己处理这些**，否则 torchrun
起 N 个进程就是 N 份互不同步的独立训练：梯度不聚合、各存各的 checkpoint、
各开一个 wandb run —— 看起来在跑，其实结果是错的。

这里提供：
    * rank / world_size / is_main 查询
    * ``wrap_ddp``      把模型包成 DDP，让 backward 自动 all-reduce 梯度
    * ``shard``         按 rank 切分训练样本，各卡跑不同数据
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
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
    return True


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(name: str = "") -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# ------------------------------------------------------------------ 模型


def wrap_ddp(model: Any) -> Any:
    """包成 DDP，让 ``loss.backward()`` 自动 all-reduce 梯度。

    ``find_unused_parameters=True``：GRPO 里一个 step 可能只有部分参数参与
    （比如某组 advantage 全 0 时），不开这个会在第二个 step 直接卡死。
    代价是一点点通信开销，稳定性优先。
    """
    if not is_distributed():
        return model
    from torch.nn.parallel import DistributedDataParallel

    device_ids = [local_rank()] if torch.cuda.is_available() else None
    return DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=local_rank() if torch.cuda.is_available() else None,
        find_unused_parameters=True,
    )


def unwrap(model: Any) -> Any:
    """取回底层模型（存盘、generate 都要用没包 DDP 的那个）。"""
    return getattr(model, "module", model)


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


def all_gather_object(obj: Any) -> list[Any]:
    if not is_distributed():
        return [obj]
    buckets: list[Any] = [None] * world_size()
    dist.all_gather_object(buckets, obj)
    return buckets


def summary_line() -> str:
    if not is_distributed():
        return "单卡"
    return f"多卡 DDP：rank {rank()}/{world_size()} (local_rank={local_rank()})"
