"""GRPO —— MiniOneRec 和 Rec-R1 共用。

True B×G prompt batching, ref-model sync, RuntimeContext AMP/strategy wiring.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import torch

from llm4rec.core import distributed as dist_utils
from llm4rec.core.exceptions import ConfigurationError
from llm4rec.core.modes import get_mode
from llm4rec.trainers.logprobs import batched_multi_prompt_logprobs, batched_sequence_logprobs, sequence_logprobs
from llm4rec.trainers.metrics_dist import reduce_reward_stats, reduce_scalar_pack
from llm4rec.trainers.ref_sync import maybe_sync_reference_model


@dataclass
class Rollout:
    prompt_ids: torch.Tensor
    completion_ids: list[torch.Tensor]
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
    r = torch.tensor(list(rewards), dtype=torch.float32)
    if normalization == "none":
        return r
    centered = r - r.mean()
    if normalization == "group_mean":
        return centered
    std = r.std(unbiased=False)
    if std < 1e-6:
        return torch.zeros_like(r)
    return centered / (std + 1e-8)


def batched_group_advantages(
    rewards_by_prompt: Sequence[Sequence[float]],
    normalization: str = "group",
) -> torch.Tensor:
    """Normalize advantages within each prompt's G group; return flat ``[B*G]``."""
    chunks = [group_advantages(r, normalization) for r in rewards_by_prompt if r]
    if not chunks:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def low_var_kl(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    diff = ref_logp - logp
    return (diff.exp() - diff - 1.0).mean()


def compute_grpo_loss(
    policy_logps: Sequence[torch.Tensor],
    ref_logps: Sequence[torch.Tensor],
    advantages: torch.Tensor,
    *,
    beta: float,
    clip_eps: float,
    kl_type: str,
) -> tuple[torch.Tensor, list[float]]:
    """GRPO clipped policy loss + optional KL. Semantics match the sequential path."""
    losses = []
    kls: list[float] = []
    for logp, ref_logp, advantage in zip(policy_logps, ref_logps, advantages, strict=True):
        if logp.numel() == 0:
            continue
        old_logp = logp.detach()
        ratio = (logp - old_logp).exp()
        adv = advantage.to(logp.device)
        unclipped = ratio * adv
        clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv
        policy_loss = -torch.min(unclipped, clipped).mean()
        if beta > 0:
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
        return torch.tensor(0.0), kls
    return torch.stack(losses).mean(), kls


def _pad_to_multiple(cfg: dict[str, Any]) -> int | None:
    opt = cfg.get("optimization") or {}
    gen = opt.get("generation") or {}
    v = gen.get("pad_to_multiple_of")
    return int(v) if v not in (None, 0, "null", False) else None


def _maybe_activation_checkpoint(
    model: Any,
    runtime: Any,
    *,
    preferred_micro: int = 1,
    selected_micro: int = 1,
) -> bool:
    from llm4rec.runtime.activation_ckpt import resolve_activation_checkpointing

    hw = (runtime.cfg.get("hardware") or {}) if runtime is not None else {}
    pressure = getattr(runtime.strategy, "pressure_ratio", None)
    if pressure is None and isinstance(hw.get("_memory_estimate"), dict):
        pressure = hw["_memory_estimate"].get("pressure_ratio")
    enable, reason = resolve_activation_checkpointing(
        hw,
        preferred_micro=preferred_micro,
        selected_micro=selected_micro,
        pressure_ratio=pressure,
        effective_strategy=getattr(runtime, "effective_strategy", None),
        strategy_source=getattr(getattr(runtime, "strategy", None), "source", None),
    )
    if enable and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    runtime.cfg.setdefault("hardware", {})["_activation_checkpointing_effective"] = enable
    runtime.cfg.setdefault("hardware", {})["_activation_checkpointing"] = {
        "requested": hw.get("activation_checkpointing", "auto"),
        "effective": enable,
        "reason": reason,
    }
    return enable


def _probe_lengths(
    train_examples: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    max_prompt_length: int,
    max_completion: int,
) -> tuple[int, int]:
    """Representative prompt/completion lengths for memory probes."""
    lengths: list[int] = []
    for ex in list(train_examples)[:128]:
        prompt = ex.get("prompt")
        try:
            if isinstance(prompt, str):
                ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            else:
                ids = tokenizer.apply_chat_template(
                    prompt, add_generation_prompt=True, tokenize=True
                )
                if hasattr(ids, "input_ids"):
                    ids = ids["input_ids"]
            lengths.append(len(ids))
        except Exception:  # noqa: BLE001
            continue
    if lengths:
        lengths.sort()
        p90 = lengths[min(len(lengths) - 1, int(0.9 * (len(lengths) - 1)))]
        prompt_len = max(32, min(int(max_prompt_length), int(p90)))
    else:
        prompt_len = min(int(max_prompt_length), 256)
    comp_len = max(4, int(max_completion))
    return prompt_len, comp_len


def _build_scoring_probe(
    model: Any,
    *,
    ref_model: Any | None,
    group_size: int,
    prompt_len: int,
    comp_len: int,
    pad_id: int | None,
    pad_mult: int | None,
    runtime: Any,
    beta: float = 0.0,
):
    """Return a probe_fn(micro_batch) that does policy+ref scoring+backward (no generate)."""

    def probe(micro_b: int) -> None:
        device = next(model.parameters()).device
        vocab_safe = 8
        prompt = torch.ones(prompt_len, dtype=torch.long, device=device) % vocab_safe + 1
        comps = [
            torch.ones(comp_len, dtype=torch.long, device=device) % vocab_safe + 1
            for _ in range(micro_b * group_size)
        ]
        prompts = [prompt for _ in comps]
        model.train()
        with runtime.autocast():
            policy_lps = batched_multi_prompt_logprobs(
                model, prompts, comps, pad_token_id=pad_id, pad_to_multiple_of=pad_mult
            )
            if ref_model is not None and float(beta) > 0.0:
                with torch.no_grad():
                    _ = batched_multi_prompt_logprobs(
                        ref_model,
                        prompts,
                        comps,
                        pad_token_id=pad_id,
                        pad_to_multiple_of=pad_mult,
                    )
            loss = torch.stack([lp.sum() for lp in policy_lps if lp.numel()]).mean()
        runtime.backward(loss)
        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return probe


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
    runtime: Any = None,
) -> dict[str, Any]:
    rl_cfg = (cfg.get("train") or {}).get(stage) or {}
    grpo_cfg = rl_cfg.get("grpo") or {}
    if not train_examples:
        raise ConfigurationError(f"{stage} 训练集为空")

    group_size = int(grpo_cfg.get("group_size") or 8)
    if group_size < 2:
        raise ConfigurationError("grpo.group_size 必须 >= 2")
    beta = float(grpo_cfg.get("beta") or 0.0)
    clip_eps = float(grpo_cfg.get("clip_epsilon") or 0.2)
    normalization = str(grpo_cfg.get("advantage_normalization") or "group")
    kl_type = str(grpo_cfg.get("kl_loss_type") or "k1")
    logging_steps = int(rl_cfg.get("logging_steps") or 1)
    sync_ref = bool(grpo_cfg.get("sync_ref_model", False))
    mixup_alpha = float(grpo_cfg.get("ref_model_mixup_alpha") or 0.6)
    sync_steps = int(grpo_cfg.get("ref_model_sync_steps") or 512)
    pad_mult = _pad_to_multiple(cfg)

    if runtime is None:
        from llm4rec.runtime.context import build_runtime

        runtime = build_runtime(cfg, log=logger.info)

    # Bind model size for strategy re-resolve before wrap
    runtime.bind_model_params(
        model,
        stage=stage if stage in {"rl", "grpo"} else "grpo",
        has_reference_model=ref_model is not None and beta > 0.0,
    )
    ref_model_strategy = runtime.resolve_reference_model_strategy()

    preferred_micro = int(
        rl_cfg.get("preferred_per_device_batch_size")
        or rl_cfg.get("per_device_batch_size")
        or 1
    )
    hw_cfg = cfg.get("hardware") or {}
    memory_auto = str(hw_cfg.get("memory") or "").lower() == "auto"
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    max_prompt_length = int(rl_cfg.get("max_prompt_length") or 512)
    # SID completions are short; use levels+8 as realistic, else cfg max_new_tokens
    max_comp = int(grpo_cfg.get("max_completion_length") or grpo_cfg.get("max_new_tokens") or 32)
    prompt_probe, comp_probe = _probe_lengths(
        train_examples,
        tokenizer,
        max_prompt_length=max_prompt_length,
        max_completion=max_comp,
    )
    micro_reduced = False

    if memory_auto:
        from llm4rec.runtime.memory import auto_tune_micro_batch

        probe = _build_scoring_probe(
            model,
            ref_model=ref_model if beta > 0.0 else None,
            group_size=group_size,
            prompt_len=prompt_probe,
            comp_len=comp_probe,
            pad_id=pad_id,
            pad_mult=pad_mult,
            runtime=runtime,
            beta=beta,
        )
        try:
            micro_b, _ = auto_tune_micro_batch(
                preferred=preferred_micro,
                world_size=runtime.world_size,
                global_batch_size=rl_cfg.get("global_batch_size")
                or rl_cfg.get("target_global_batch_size")
                or rl_cfg.get("reference_global_batch_size"),
                mode=runtime.mode,
                memory_auto=True,
                probe_fn=probe,
                batch_policy=hw_cfg.get("batch_policy"),
                log=logger.info,
            )
            micro_reduced = micro_b < preferred_micro
            rl_cfg["per_device_batch_size"] = micro_b
            logger.info(
                f"[memory-auto] preferred={preferred_micro} selected={micro_b} "
                f"probe_prompt={prompt_probe} probe_comp={comp_probe} G={group_size} "
                f"ref_scoring={ref_model is not None and beta > 0.0}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.info(f"[memory-auto] probe skipped ({exc}); using preferred={preferred_micro}")

    batch_plan = runtime.resolve_stage_batch(stage, rl_cfg)
    accum = batch_plan.gradient_accumulation_steps
    per_device_b = batch_plan.per_device_batch_size
    if dist_utils.is_main():
        for line in batch_plan.summary_lines():
            logger.info(f"[{stage}] {line}")

    local_examples = dist_utils.shard(train_examples)
    if not local_examples:
        raise ConfigurationError(
            f"rank {dist_utils.rank()} 分到 0 条样本 —— 训练集比卡数还少"
        )

    max_steps = rl_cfg.get("max_steps")
    if max_steps in (None, 0, "null"):
        epochs = float(rl_cfg.get("epochs") or 1)
        # Each optimizer step consumes B prompts × accum micro-steps
        steps_per_epoch = max(1, math.ceil(len(local_examples) / max(per_device_b, 1) / accum))
        max_steps = int(math.ceil(steps_per_epoch * epochs))
    max_steps = int(max_steps)

    from llm4rec.runtime.checkpointing import (
        resolve_save_steps,
        resolve_save_total_limit,
        save_step_checkpoint,
        should_save_at_step,
    )

    save_every = resolve_save_steps(cfg, rl_cfg, max_steps=max_steps, as_int=True)
    if save_every is not None:
        logger.info(
            f"[{stage}] 中间 checkpoint：每 {save_every} step 存一次"
            f"（最多保留 {resolve_save_total_limit(cfg)} 个）"
        )

    _maybe_activation_checkpoint(
        model,
        runtime,
        preferred_micro=preferred_micro,
        selected_micro=per_device_b,
    )

    base_model = model
    # Compile → wrap → optimizer(trainable wrapped params)
    train_core = runtime.maybe_compile(model, name="grpo_policy")
    train_model = runtime.wrap_model(train_core)
    from llm4rec.trainers.schedulers import build_optimizer, create_scheduler

    optimizer, optim_fallback = build_optimizer(
        [p for p in train_model.parameters() if p.requires_grad],
        lr=float(rl_cfg.get("learning_rate") or 1e-6),
        betas=(float(rl_cfg.get("adam_beta1") or 0.9), float(rl_cfg.get("adam_beta2") or 0.999)),
        eps=float(rl_cfg.get("adam_epsilon") or 1e-8),
        weight_decay=float(rl_cfg.get("weight_decay") or 0.0),
        optim_name=rl_cfg.get("optim") or rl_cfg.get("optimizer"),
    )
    if optim_fallback and dist_utils.is_main():
        logger.info(f"[grpo] optimizer fallback: {optim_fallback}")
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=str(rl_cfg.get("lr_scheduler_type") or "constant"),
        num_training_steps=max_steps,
        warmup_ratio=rl_cfg.get("warmup_ratio"),
        warmup_steps=rl_cfg.get("warmup_steps"),
    )
    max_grad_norm = float(rl_cfg.get("max_grad_norm") or 1.0)
    is_main = dist_utils.is_main()

    if ref_model is not None:
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        # Memory-aware reference model placement
        if ref_model_strategy == "cpu_offload" and torch.cuda.is_available():
            try:
                ref_model.to("cpu")
            except Exception:
                pass
        elif ref_model_strategy == "fsdp":
            try:
                from llm4rec.core import distributed as _dist

                ref_model = _dist.wrap_fsdp(
                    ref_model,
                    param_dtype=None if runtime.precision.precision == "fp32" else runtime.dtype,
                    reduce_dtype=None if runtime.precision.precision == "fp32" else runtime.dtype,
                    buffer_dtype=None if runtime.precision.precision == "fp32" else runtime.dtype,
                )
            except Exception as exc:
                logger.info(f"[grpo] ref FSDP wrap skipped ({exc}); keeping replicated")

    from llm4rec.runtime.profiler import make_timer, peak_vram_gb

    timer = make_timer(cfg)
    throughput: dict[str, float] = {}

    state = GRPOState()
    base_model.train()
    logger.info(
        f"[{stage}] GRPO：steps={max_steps} B={per_device_b} G={group_size} beta={beta} "
        f"clip={clip_eps} accum={accum} precision={runtime.precision.precision} "
        f"strategy={runtime.effective_strategy} sync_ref={sync_ref} "
        f"scheduler={rl_cfg.get('lr_scheduler_type')} "
        f"n_train={len(train_examples)} (rank {len(local_examples)})  {dist_utils.summary_line()}"
    )

    cursor = 0
    micro = 0
    optimizer.zero_grad()

    from llm4rec.runtime.profiler import make_scheduled_profiler

    profiler = make_scheduled_profiler(
        cfg, output_dir=str(Path(output_dir) / "profiler"), rank=dist_utils.rank()
    )
    if profiler is not None:
        profiler.start()

    while state.step < max_steps:
        batch_examples = []
        for _ in range(per_device_b):
            batch_examples.append(local_examples[cursor % len(local_examples)])
            cursor += 1

        rollouts: list[Rollout] = []
        with timer.phase("generation"):
            with torch.no_grad():
                for example in batch_examples:
                    rollouts.append(rollout_fn(base_model, tokenizer, example, group_size))
        # Drop empty groups but keep shape alignment via filtering
        valid = [r for r in rollouts if r.completion_ids]
        if not valid:
            continue

        rewards_by_prompt = [reward_fn(r) for r in valid]
        advantages = batched_group_advantages(rewards_by_prompt, normalization)
        flat_rewards = [rw for group in rewards_by_prompt for rw in group]

        prompts: list[torch.Tensor] = []
        comps: list[torch.Tensor] = []
        for r in valid:
            for c in r.completion_ids:
                prompts.append(r.prompt_ids)
                comps.append(c)

        with runtime.autocast():
            with timer.phase("policy_scoring"):
                policy_logps = batched_multi_prompt_logprobs(
                    train_model,
                    prompts,
                    comps,
                    pad_token_id=pad_id,
                    pad_to_multiple_of=pad_mult,
                )
            with torch.no_grad():
                with timer.phase("reference_scoring"):
                    if ref_model is not None and beta > 0:
                        ref_logps = batched_multi_prompt_logprobs(
                            ref_model,
                            prompts,
                            comps,
                            pad_token_id=pad_id,
                            pad_to_multiple_of=pad_mult,
                        )
                    else:
                        ref_logps = [lp.detach() for lp in policy_logps]

            with timer.phase("loss"):
                loss_raw, kls = compute_grpo_loss(
                    policy_logps,
                    ref_logps,
                    advantages,
                    beta=beta,
                    clip_eps=clip_eps,
                    kl_type=kl_type,
                )
            if not any(lp.numel() > 0 and lp.requires_grad for lp in policy_logps):
                continue
            loss = loss_raw / accum

        is_sync_step = (micro + 1) % accum == 0
        with timer.phase("backward"):
            if is_sync_step:
                runtime.backward(loss)
            else:
                with dist_utils.no_sync(train_model):
                    runtime.backward(loss)
        micro += 1

        if micro % accum != 0:
            continue

        with timer.phase("optimizer"):
            runtime.optimizer_step(
                optimizer,
                parameters=[p for p in train_model.parameters() if p.requires_grad],
                max_grad_norm=max_grad_norm,
            )
            optimizer.zero_grad()
            scheduler.step()
        if profiler is not None:
            profiler.step()
        state.step += 1

        maybe_sync_reference_model(
            train_model,
            ref_model,
            state.step,
            enabled=sync_ref,
            alpha=mixup_alpha,
            sync_steps=sync_steps,
        )

        # Every rank evaluates the same logging condition; pack collectives.
        should_log = state.step % logging_steps == 0
        if should_log:
            reward_stats = reduce_reward_stats(flat_rewards)
            mean_kl = reduce_scalar_pack([sum(kls) / len(kls) if kls else 0.0])[0]
            mean_loss = reduce_scalar_pack([float(loss_raw.detach())])[0]
            mean_adv = reduce_scalar_pack([float(advantages.abs().mean()) if advantages.numel() else 0.0])[0]
            state.last_reward = reward_stats["reward"]
            state.last_kl = mean_kl
            if is_main:
                metrics = {
                    "loss": mean_loss,
                    "reward": reward_stats["reward"],
                    "reward_max": reward_stats["reward_max"],
                    "reward_min": reward_stats["reward_min"],
                    "reward_std": reward_stats["reward_std"],
                    "kl": mean_kl,
                    "advantage_abs_mean": mean_adv,
                    "per_device_batch": float(per_device_b),
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                if timer.enabled or state.step == 1 or state.step == max_steps:
                    phase = timer.summary()
                    if phase:
                        metrics["phase_ms"] = {
                            k: round(v["mean_s"] * 1000, 2) for k, v in phase.items()
                        }
                    vram = peak_vram_gb()
                    if vram is not None:
                        metrics["peak_vram_gb"] = vram
                        throughput["peak_vram_gb"] = vram
                state.history.append({"step": state.step, **metrics})
                logger.log_metrics(
                    metrics,
                    stage=stage,
                    step=state.step,
                    split="train",
                    wandb_prefix="train",
                )
        else:
            # Keep last_* updated cheaply without extra collectives when not logging
            if flat_rewards:
                state.last_reward = float(sum(flat_rewards) / len(flat_rewards))
            if kls:
                state.last_kl = float(sum(kls) / len(kls))

        for callback in callbacks:
            hook = getattr(callback, "on_step", None)
            if hook is not None:
                hook(state.step, base_model)

        if should_save_at_step(state.step, save_every):
            save_step_checkpoint(
                train_model,
                output_dir,
                state.step,
                tokenizer=tokenizer,
                cfg=cfg,
                logger=logger,
                tag=stage,
            )

    for callback in callbacks:
        hook = getattr(callback, "on_train_end", None)
        if hook is not None:
            hook(state.step, base_model)

    if profiler is not None:
        try:
            profiler.stop()
        except Exception:  # noqa: BLE001
            pass

    dist_utils.barrier("grpo_end")
    final_dir = Path(output_dir) / "final"
    dist_utils.save_pretrained_distributed(train_model, final_dir, tokenizer=tokenizer, is_main=is_main)
    if is_main:
        (Path(output_dir) / "train_log.json").write_text(
            json.dumps(state.history, indent=2) + "\n", encoding="utf-8"
        )
    dist_utils.barrier("grpo_saved")

    logger.info(f"[{stage}] 完成 {state.step} steps，权重 → {final_dir}")
    phase = timer.summary()
    # Windowed rates from total phase times when available
    total_s = sum(v.get("total_s", 0.0) for v in phase.values()) if phase else 0.0
    prompts_total = int(state.step * per_device_b * accum)
    seq_total = int(prompts_total * group_size)
    perf: dict[str, Any] = {}
    if total_s > 0:
        perf["prompts_per_sec"] = round(prompts_total / total_s, 3)
        perf["rollout_sequences_per_sec"] = round(seq_total / total_s, 3)
    if throughput.get("peak_vram_gb") is not None:
        perf["peak_vram_gb"] = throughput["peak_vram_gb"]
    if phase:
        perf["phase_timers"] = phase
    if perf:
        cfg.setdefault("_performance", {})["grpo"] = perf
    return {
        "stage": stage,
        "checkpoint": str(final_dir),
        "steps": state.step,
        "last_reward": state.last_reward,
        "last_kl": state.last_kl,
        "mode": get_mode(cfg),
        "precision": runtime.precision.precision,
        "strategy": runtime.effective_strategy,
        "reference_model_strategy": ref_model_strategy,
        "batch_plan": batch_plan.to_dict(),
        "optimizer_fallback_reason": optim_fallback,
        "scheduler": str(rl_cfg.get("lr_scheduler_type") or "constant"),
        "warmup_ratio": rl_cfg.get("warmup_ratio"),
        "throughput": throughput,
        "phase_timers": phase,
        "performance": perf or None,
    }


__all__ = [
    "Rollout",
    "RolloutFn",
    "RewardFn",
    "GRPOState",
    "group_advantages",
    "batched_group_advantages",
    "sequence_logprobs",
    "batched_sequence_logprobs",
    "batched_multi_prompt_logprobs",
    "low_var_kl",
    "compute_grpo_loss",
    "run_grpo",
]
