"""GRPO —— MiniOneRec 和 Rec-R1 共用。

两条路线的 RL 算法完全相同（都是 GRPO，无 critic），差别只在：
  * 怎么采样      —— MiniOneRec 走约束 beam，Rec-R1 走普通 sampling
  * reward 怎么算 —— MiniOneRec 是 SID 命中 + 位次，Rec-R1 是检索 NDCG + 格式

这两处都通过 ``Rollout`` / ``RewardFn`` 注入，训练循环本身共享。
这样"RL 算法"就不再是路线之间的混杂变量，bias 差异可以归因到任务形态。

GRPO 的核心（相对 PPO 省掉 critic）：
    对每个 prompt 采 G 条 → 组内标准化 reward 当 advantage
    A_i = (r_i - mean(r)) / (std(r) + eps)
    loss = -E[min(ratio * A, clip(ratio, 1±eps) * A)] + beta * KL(pi || pi_ref)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import torch
import torch.nn.functional as F

from llm4rec.core import distributed as dist_utils
from llm4rec.core.exceptions import ConfigurationError


@dataclass
class Rollout:
    """一个 prompt 的一组采样结果。"""

    prompt_ids: torch.Tensor          # [prompt_len]
    completion_ids: list[torch.Tensor]  # G 条，每条 [comp_len]
    texts: list[str]
    example: dict[str, Any]


class RolloutFn(Protocol):
    def __call__(
        self, model: Any, tokenizer: Any, example: dict[str, Any], group_size: int
    ) -> Rollout: ...


RewardFn = Callable[[Rollout], list[float]]


@dataclass
class GRPOState:
    step: int = 0
    last_reward: float = 0.0
    last_kl: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)


def group_advantages(rewards: Sequence[float], normalization: str = "group") -> torch.Tensor:
    """组内标准化 —— GRPO 用它替代 critic 估计基线。"""
    r = torch.tensor(list(rewards), dtype=torch.float32)
    if normalization == "none":
        return r
    centered = r - r.mean()
    if normalization == "group_mean":
        return centered
    std = r.std(unbiased=False)
    if std < 1e-6:
        # 组内 reward 全一样 → advantage 全 0，这一组不产生梯度。
        # 这是正常现象（比如全对或全错），不是 bug。
        return torch.zeros_like(r)
    return centered / (std + 1e-8)


def sequence_logprobs(
    model: Any,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
) -> torch.Tensor:
    """算 completion 每个 token 的 log prob（对 prompt 部分不算）。"""
    full = torch.cat([prompt_ids, completion_ids]).unsqueeze(0)
    logits = model(full).logits[0]
    # 预测第 t 个 token 用的是第 t-1 个位置的 logits
    start = prompt_ids.shape[0] - 1
    end = full.shape[1] - 1
    target_logits = logits[start:end]
    log_probs = F.log_softmax(target_logits.float(), dim=-1)
    return log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)


def low_var_kl(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """k3 估计量（Rec-R1 的 ``kl_loss_type=low_var_kl``）。

    比朴素的 (logp - ref_logp) 方差小很多，且恒为非负。
    """
    diff = ref_logp - logp
    return (diff.exp() - diff - 1.0).mean()


def run_grpo(
    *,
    cfg: dict[str, Any],
    model: Any,
    ref_model: Any,
    tokenizer: Any,
    train_examples: Sequence[dict[str, Any]],
    rollout_fn: RolloutFn,
    reward_fn: RewardFn,
    output_dir: Path,
    logger: Any,
    callbacks: Sequence[Any] = (),
    stage: str = "rl",
) -> dict[str, Any]:
    """GRPO 主循环。

    ``callbacks`` 里的对象只要有 ``on_step(step, model)`` 就会被调用 ——
    在线 bias 评测就是从这里挂进去的。
    """
    rl_cfg = (cfg.get("train") or {}).get(stage) or {}
    grpo_cfg = rl_cfg.get("grpo") or {}
    if not train_examples:
        raise ConfigurationError(f"{stage} 训练集为空")

    group_size = int(grpo_cfg.get("group_size") or 8)
    if group_size < 2:
        raise ConfigurationError("grpo.group_size 必须 >= 2（组内标准化需要至少两条）")
    beta = float(grpo_cfg.get("beta") or 0.0)
    clip_eps = float(grpo_cfg.get("clip_epsilon") or 0.2)
    normalization = str(grpo_cfg.get("advantage_normalization") or "group")
    kl_type = str(grpo_cfg.get("kl_loss_type") or "k1")
    accum = int(rl_cfg.get("gradient_accumulation_steps") or 1)
    logging_steps = int(rl_cfg.get("logging_steps") or 1)

    # —— 多卡：各 rank 跑不同样本，梯度由 DDP 在 backward 里 all-reduce ——
    local_examples = dist_utils.shard(train_examples)
    if not local_examples:
        raise ConfigurationError(
            f"rank {dist_utils.rank()} 分到 0 条样本 —— 训练集比卡数还少"
        )

    max_steps = rl_cfg.get("max_steps")
    if max_steps in (None, 0, "null"):
        epochs = float(rl_cfg.get("epochs") or 1)
        # 用本 rank 的样本数算：各卡跑一样多的 step，才不会有人先结束卡住 barrier
        max_steps = int(math.ceil(len(local_examples) * epochs / accum))
    max_steps = int(max_steps)

    # 优化器建在【未包 DDP】的参数上，state_dict 才和单卡格式一致
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(rl_cfg.get("learning_rate") or 1e-6),
        betas=(float(rl_cfg.get("adam_beta1") or 0.9), float(rl_cfg.get("adam_beta2") or 0.999)),
        eps=float(rl_cfg.get("adam_epsilon") or 1e-8),
        weight_decay=float(rl_cfg.get("weight_decay") or 0.0),
    )
    max_grad_norm = float(rl_cfg.get("max_grad_norm") or 1.0)

    base_model = model                       # 采样 / 存盘用它（generate 不能走 DDP）
    train_model = dist_utils.wrap_ddp(model)  # 反向传播走它
    is_main = dist_utils.is_main()

    state = GRPOState()
    base_model.train()
    if ref_model is not None:
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    logger.info(
        f"[{stage}] GRPO 开始：steps={max_steps} G={group_size} beta={beta} "
        f"clip={clip_eps} accum={accum} n_train={len(train_examples)} "
        f"(本 rank {len(local_examples)})  {dist_utils.summary_line()}"
    )

    cursor = 0
    micro = 0
    optimizer.zero_grad()

    while state.step < max_steps:
        example = local_examples[cursor % len(local_examples)]
        cursor += 1

        # ---- 1) 采样一组（用未包 DDP 的模型，generate 走 DDP 会挂）----
        with torch.no_grad():
            rollout = rollout_fn(base_model, tokenizer, example, group_size)
        if not rollout.completion_ids:
            continue

        # ---- 2) 算 reward ----
        rewards = reward_fn(rollout)
        advantages = group_advantages(rewards, normalization)

        # ---- 3) 策略梯度 ----
        losses = []
        kls = []
        for completion, advantage in zip(rollout.completion_ids, advantages, strict=True):
            if completion.numel() == 0:
                continue
            logp = sequence_logprobs(train_model, rollout.prompt_ids, completion)

            with torch.no_grad():
                old_logp = logp.detach()
                ref_logp = (
                    sequence_logprobs(ref_model, rollout.prompt_ids, completion)
                    if ref_model is not None
                    else old_logp
                )

            # updates_per_rollout=1 时 ratio 恒为 1，clip 不生效 —— 这是
            # on-policy 的正常情形，保留 clip 是为了支持 >1 的配置。
            ratio = (logp - old_logp).exp()
            adv = advantage.to(logp.device)
            unclipped = ratio * adv
            clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()

            if beta > 0 and ref_model is not None:
                kl = (
                    low_var_kl(logp, ref_logp)
                    if kl_type == "low_var_kl"
                    else (logp - ref_logp).mean()
                )
                losses.append(policy_loss + beta * kl)
                kls.append(float(kl.detach()))
            else:
                losses.append(policy_loss)

        if not losses:
            continue

        loss = torch.stack(losses).mean() / accum

        # 梯度累积的中间步跳过 all-reduce，只在最后一个 micro-step 同步。
        # 不这么做的话 accum=8 会白白多做 7 次全量梯度通信。
        is_sync_step = (micro + 1) % accum == 0
        if is_sync_step:
            loss.backward()
        else:
            with dist_utils.no_sync(train_model):
                loss.backward()
        micro += 1

        if micro % accum != 0:
            continue

        torch.nn.utils.clip_grad_norm_(
            [p for p in base_model.parameters() if p.requires_grad], max_grad_norm
        )
        optimizer.step()
        optimizer.zero_grad()
        state.step += 1

        # 日志里的 loss/reward 要是【全局】均值，不是本卡的
        mean_reward = dist_utils.all_reduce_mean(sum(rewards) / len(rewards))
        mean_kl = dist_utils.all_reduce_mean(sum(kls) / len(kls) if kls else 0.0)
        state.last_reward, state.last_kl = mean_reward, mean_kl

        if state.step % logging_steps == 0 and is_main:
            metrics = {
                "loss": dist_utils.all_reduce_mean(float(loss.detach()) * accum),
                "reward": mean_reward,
                "reward_max": float(max(rewards)),
                "reward_min": float(min(rewards)),
                "reward_std": float(torch.tensor(rewards).std(unbiased=False)),
                "kl": mean_kl,
                "advantage_abs_mean": float(advantages.abs().mean()),
            }
            state.history.append({"step": state.step, **metrics})
            logger.log_metrics(
                metrics,
                stage=stage,
                step=state.step,
                split="train",
                wandb_prefix="train",
            )

        # 在线 bias 评测：所有 rank 都要进（内部按 rank 分片再 all-gather），
        # 只让 rank0 进会让其它 rank 空转在下一次 all-reduce 上，看起来像卡死。
        for callback in callbacks:
            hook = getattr(callback, "on_step", None)
            if hook is not None:
                hook(state.step, base_model)

    # ---- 收尾 ----
    for callback in callbacks:
        hook = getattr(callback, "on_train_end", None)
        if hook is not None:
            hook(state.step, base_model)

    dist_utils.barrier("grpo_end")
    final_dir = Path(output_dir) / "final"
    if is_main:
        final_dir.mkdir(parents=True, exist_ok=True)
        dist_utils.unwrap(base_model).save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        (Path(output_dir) / "train_log.json").write_text(
            json.dumps(state.history, indent=2) + "\n", encoding="utf-8"
        )
    # 其它 rank 等 rank0 写完再往下走，否则下一个 stage 会读到半截 checkpoint
    dist_utils.barrier("grpo_saved")

    logger.info(f"[{stage}] 完成 {state.step} steps，权重 → {final_dir}")
    return {
        "stage": stage,
        "checkpoint": str(final_dir),
        "steps": state.step,
        "last_reward": state.last_reward,
        "last_kl": state.last_kl,
    }
