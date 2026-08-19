"""RL 训练中的在线 bias 评测。

为什么要在线评：我们的假设是 **RL 会放大 bias**，所以需要 bias 随 RL step
的漂移曲线，而不是只有首尾两个点。但按 step 存 checkpoint 事后再评的话，
0.5B 全参一份 ~2GB，跑 500 步存 10 份就 20GB —— 不现实。

所以改成：训练循环里每 N 步，对一个**固定的** held-out 子集跑一次解码，
当场算 bias 直接推 wandb，**一份 checkpoint 都不用多存**。

子集用固定 seed 采样并且全程不变，所以不同 step 之间的曲线是可比的。

多卡：把子集按 rank 切片，各自解码后写盘汇总（不用 NCCL all_gather）。
不让 rank0 单独评（其它 rank 干等在 barrier 里会看起来像卡死）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from llm4rec.decoders.base import Decoder
from llm4rec.eval.bias import RankedResult, compute_bias_metrics
from llm4rec.eval.catalog import ItemCatalog
from llm4rec.eval.gather import gather_ranked_results


@dataclass
class OnlineBiasEvaluator:
    """对固定 held-out 子集算 bias。stage 内多次调用之间完全可比。"""

    decoder: Decoder
    catalog: ItemCatalog
    examples: list[dict[str, Any]]
    top_k: int = 10
    ips_gamma: float = 1.0
    tier_thresholds: dict[str, float] | None = None
    enabled_metrics: list[str] | None = None
    shard_dir: Path | None = None
    gather_timeout_s: float = 8 * 3600

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        *,
        decoder: Decoder,
        catalog: ItemCatalog,
        pool: Sequence[dict[str, Any]],
        n_examples: int | None = None,
        seed: int = 42,
        shard_dir: Path | None = None,
    ) -> OnlineBiasEvaluator:
        bias_cfg = cfg.get("bias") or {}
        n = int(n_examples or bias_cfg.get("online_examples") or 256)
        rng = random.Random(seed)
        items = list(pool)
        # 固定 seed 采样：整个 stage 用同一批样本，曲线才有意义
        subset = items if len(items) <= n else rng.sample(items, n)
        return cls(
            decoder=decoder,
            catalog=catalog,
            examples=subset,
            top_k=int(bias_cfg.get("top_k") or 10),
            ips_gamma=float(bias_cfg.get("ips_gamma") or 1.0),
            tier_thresholds=dict(bias_cfg.get("tiers") or {}) or None,
            enabled_metrics=list(bias_cfg.get("metrics") or []) or None,
            shard_dir=Path(shard_dir) if shard_dir is not None else None,
            gather_timeout_s=float(bias_cfg.get("gather_timeout_s") or 8 * 3600),
        )

    def evaluate(
        self, model: Any, tokenizer: Any, *, name: str = "online"
    ) -> dict[str, Any]:
        shard = _shard_for_rank(self.examples)
        was_training = bool(getattr(model, "training", False))
        model.eval()
        try:
            with torch.no_grad():
                progress_dir = None
                if self.shard_dir is not None:
                    progress_dir = Path(self.shard_dir).parent / "progress"
                local = self.decoder.decode_batch(
                    model,
                    tokenizer,
                    shard,
                    top_k=self.top_k,
                    progress_total=len(self.examples),
                    progress_dir=progress_dir,
                    progress_name=name,
                )
        finally:
            if was_training:
                model.train()

        results = _gather_results(
            local, shard_dir=self.shard_dir, name=name, timeout_s=self.gather_timeout_s
        )
        return compute_bias_metrics(
            results,
            self.catalog,
            top_k=self.top_k,
            ips_gamma=self.ips_gamma,
            tier_thresholds=self.tier_thresholds,
            enabled=self.enabled_metrics,
        )


class BiasEvalCallback(TrainerCallback):
    """挂在 HF Trainer 上，每 N 步在线评一次 bias 并推 wandb。

    只在 ``bias.online_stages`` 里列出的 stage 生效（默认 rl / dpo，不含 sft）。
    """

    def __init__(
        self,
        evaluator: OnlineBiasEvaluator,
        logger: Any,
        *,
        stage: str,
        every_n_steps: int = 50,
        tokenizer: Any = None,
    ) -> None:
        self.evaluator = evaluator
        self.logger = logger
        self.stage = stage
        self.every_n_steps = max(1, int(every_n_steps))
        self.tokenizer = tokenizer
        self.history: list[dict[str, Any]] = []

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        step = int(state.global_step)
        if step == 0 or step % self.every_n_steps != 0:
            return
        self._run(step, kwargs.get("model"))

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        # stage 结束再评一次，保证曲线末端一定有点
        self._run(int(state.global_step), kwargs.get("model"), final=True)

    def _run(self, step: int, model: Any, *, final: bool = False) -> None:
        if model is None:
            return
        tok = self.tokenizer
        try:
            metrics = self.evaluator.evaluate(
                model, tok, name=f"{self.stage}_step{step}"
            )
        except Exception as exc:  # noqa: BLE001
            # 在线评测挂了不能把训练带崩 —— 记一笔继续训
            self.logger.warning(f"[bias] step {step} 在线评测失败：{exc}")
            return

        metrics["step"] = float(step)
        self.history.append(dict(metrics))
        self.logger.log_metrics(
            {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            stage=self.stage,
            step=step,
            split="bias_online",
            wandb_prefix="bias",
        )
        if final:
            self.logger.info(f"[bias] {self.stage} 结束 @ step {step}: {_brief(metrics)}")


def _brief(metrics: dict[str, Any]) -> str:
    keys = ["pop_lift@1", "delta_gap", "exposure_gini", "tier_gap", "history_copy_rate"]
    parts = [f"{k}={metrics[k]:.4f}" for k in keys if isinstance(metrics.get(k), float)]
    return " ".join(parts)


# ---------------------------------------------------------------- 分布式辅助


def _dist_info() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _shard_for_rank(examples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """把评测子集按 rank 切开，各卡各评一份，避免其它 rank 空转。"""
    rank, world = _dist_info()
    if world <= 1:
        return list(examples)
    return [ex for i, ex in enumerate(examples) if i % world == rank]


def _gather_results(
    local: list[RankedResult],
    *,
    shard_dir: Path | None,
    name: str,
    timeout_s: float,
) -> list[RankedResult]:
    """把各 rank 的结果收拢到一起（每张卡都拿到完整列表）。"""
    if shard_dir is None:
        rank, world = _dist_info()
        if world <= 1:
            return local
        raise RuntimeError("online eval 多卡必须提供 shard_dir，不能再用 NCCL all_gather")
    return gather_ranked_results(
        local, shard_dir, name=name, timeout_s=timeout_s
    )
