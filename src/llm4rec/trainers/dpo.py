"""DPO4Rec DPO training — true 2B preference minibatch + RuntimeContext."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

from llm4rec.core import distributed as dist_utils
from llm4rec.core.exceptions import ConfigurationError
from llm4rec.core.modes import get_mode
from llm4rec.trainers.logprobs import score_preference_batch, sequence_logprobs
from llm4rec.trainers.metrics_dist import reduce_scalar_pack


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    chosen_logratio = policy_chosen - ref_chosen
    rejected_logratio = policy_rejected - ref_rejected
    margin = beta * (chosen_logratio - rejected_logratio)
    loss = -F.logsigmoid(margin).mean()
    stats = {
        "dpo_margin": float(margin.mean()),
        "chosen_reward": float(beta * chosen_logratio.mean()),
        "rejected_reward": float(beta * rejected_logratio.mean()),
        "accuracy": float((margin > 0).float().mean()),
    }
    return loss, stats


@torch.no_grad()
def sample_reasonings(
    model: Any,
    tokenizer: Any,
    example: dict[str, Any],
    *,
    n: int,
    temperature: float,
    max_new_tokens: int,
    use_cache: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor], list[str]]:
    device = next(model.parameters()).device
    encoded = tokenizer.apply_chat_template(
        example["prompt"], add_generation_prompt=True, return_tensors="pt"
    )
    ids = (encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]).to(device)
    prompt_len = ids.shape[1]
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id

    output = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        num_return_sequences=n,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        pad_token_id=pad,
        use_cache=use_cache,
    )

    completions, texts = [], []
    for sequence in output:
        comp = sequence[prompt_len:]
        if pad is not None:
            nonpad = (comp != pad).nonzero()
            comp = comp[: int(nonpad[-1]) + 1] if nonpad.numel() else comp[:0]
        completions.append(comp)
        texts.append(tokenizer.decode(comp, skip_special_tokens=True))
    return ids[0], completions, texts


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


def run_dpo(
    *,
    cfg: dict[str, Any],
    model: Any,
    ref_model: Any,
    tokenizer: Any,
    train_examples: Sequence[dict[str, Any]],
    score_fn: Callable[[dict[str, Any], Sequence[str]], list[float]],
    output_dir: Path,
    logger: Any,
    callbacks: Sequence[Any] = (),
    on_iteration_end: Callable[[int, dict[str, str]], None] | None = None,
    runtime: Any = None,
) -> dict[str, Any]:
    dpo_cfg = (cfg.get("train") or {}).get("dpo") or {}
    if not train_examples:
        raise ConfigurationError("DPO 训练集为空")

    iterations = int(dpo_cfg.get("iterations") or 2)
    n_samples = int(dpo_cfg.get("num_samples") or 10)
    beta = float(dpo_cfg.get("beta") or 0.01)
    temperature = float(dpo_cfg.get("sampling_temperature") or 1.0)
    max_new_tokens = int(dpo_cfg.get("max_new_tokens") or 512)
    epochs = int(dpo_cfg.get("epochs") or 3)
    logging_steps = int(dpo_cfg.get("logging_steps") or 1)
    pad_mult = _pad_to_multiple(cfg)

    if runtime is None:
        from llm4rec.runtime.context import build_runtime

        runtime = build_runtime(cfg, log=logger.info)

    runtime.bind_model_params(model, stage="dpo")
    runtime.resolve_reference_model_strategy()

    preferred_micro = int(dpo_cfg.get("per_device_batch_size") or dpo_cfg.get("pair_batch_size") or 1)
    hw_cfg = cfg.get("hardware") or {}
    memory_auto = str(hw_cfg.get("memory") or "").lower() == "auto"
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    if memory_auto:
        from llm4rec.runtime.memory import auto_tune_micro_batch

        def probe(micro_b: int) -> None:
            device = next(model.parameters()).device
            prompts = [torch.ones(8, dtype=torch.long, device=device) for _ in range(micro_b)]
            ch = [torch.ones(4, dtype=torch.long, device=device) for _ in range(micro_b)]
            rj = [torch.ones(4, dtype=torch.long, device=device) for _ in range(micro_b)]
            model.train()
            with runtime.autocast():
                pc, pr = score_preference_batch(
                    model, prompts, ch, rj, pad_token_id=pad_id, pad_to_multiple_of=pad_mult
                )
                loss = (pc - pr).mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        try:
            micro_b, _ = auto_tune_micro_batch(
                preferred=preferred_micro,
                world_size=runtime.world_size,
                global_batch_size=dpo_cfg.get("global_batch_size") or dpo_cfg.get("target_global_batch_size"),
                mode=runtime.mode,
                memory_auto=True,
                probe_fn=probe,
                batch_policy=hw_cfg.get("batch_policy"),
                log=logger.info,
            )
            dpo_cfg["per_device_batch_size"] = micro_b
        except Exception as exc:  # noqa: BLE001
            logger.info(f"[memory-auto] probe skipped ({exc}); using preferred={preferred_micro}")

    batch_plan = runtime.resolve_stage_batch("dpo", dpo_cfg)
    accum = batch_plan.gradient_accumulation_steps
    pair_batch_size = batch_plan.per_device_batch_size
    if dist_utils.is_main():
        for line in batch_plan.summary_lines():
            logger.info(f"[dpo] {line}")

    local_examples = dist_utils.shard(train_examples)
    if not local_examples:
        raise ConfigurationError(
            f"rank {dist_utils.rank()} 分到 0 条样本 —— 训练集比卡数还少"
        )

    _maybe_activation_checkpoint(
        model,
        runtime,
        preferred_micro=preferred_micro,
        selected_micro=pair_batch_size,
    )

    base_model = model
    train_core = runtime.maybe_compile(model, name="dpo_policy")
    train_model = runtime.wrap_model(train_core)
    from llm4rec.trainers.schedulers import build_optimizer, create_scheduler

    optimizer, optim_fallback = build_optimizer(
        [p for p in train_model.parameters() if p.requires_grad],
        lr=float(dpo_cfg.get("learning_rate") or 5e-5),
        betas=(float(dpo_cfg.get("adam_beta1") or 0.9), float(dpo_cfg.get("adam_beta2") or 0.999)),
        eps=float(dpo_cfg.get("adam_epsilon") or 1e-8),
        weight_decay=float(dpo_cfg.get("weight_decay") or 0.0),
        optim_name=dpo_cfg.get("optim") or dpo_cfg.get("optimizer"),
    )
    if optim_fallback:
        logger.info(f"[dpo] optimizer fallback: {optim_fallback}")
    # Scheduler created after we know approx steps; use a generous upper bound then rebuild
    approx_steps = max(1, iterations * epochs * max(1, len(local_examples) // max(pair_batch_size, 1) // max(accum, 1)))
    from llm4rec.runtime.checkpointing import (
        resolve_save_steps,
        resolve_save_total_limit,
        save_step_checkpoint,
        should_save_at_step,
    )

    save_every = resolve_save_steps(cfg, dpo_cfg, max_steps=approx_steps, as_int=True)
    if save_every is not None:
        logger.info(
            f"[dpo] 中间 checkpoint：每 {save_every} step 存一次"
            f"（最多保留 {resolve_save_total_limit(cfg)} 个）"
        )
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=str(dpo_cfg.get("lr_scheduler_type") or "constant"),
        num_training_steps=approx_steps,
        warmup_ratio=dpo_cfg.get("warmup_ratio"),
        warmup_steps=dpo_cfg.get("warmup_steps"),
    )
    max_grad_norm = float(dpo_cfg.get("max_grad_norm") or 1.0)
    is_main = dist_utils.is_main()

    if ref_model is not None:
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    # KV cache for sampling: auto → try use_cache=True (dynamic); static left to HF if supported
    gen_cfg = (cfg.get("optimization") or {}).get("generation") or {}
    cache_mode = str(gen_cfg.get("cache") or "auto").lower()
    use_cache = cache_mode != "false"

    from llm4rec.runtime.profiler import make_timer, peak_vram_gb

    timer = make_timer(cfg)
    throughput: dict[str, float] = {}

    logger.info(
        f"[dpo] iterations={iterations} N={n_samples} beta={beta} epochs={epochs} "
        f"pair_bs={pair_batch_size} precision={runtime.precision.precision} "
        f"strategy={runtime.effective_strategy} n_train={len(train_examples)} "
        f"(rank {len(local_examples)})  {dist_utils.summary_line()}"
    )

    global_step = 0
    history: list[dict[str, Any]] = []
    best_reasoning: dict[str, str] = {}

    for iteration in range(1, iterations + 1):
        logger.info(f"[dpo] 迭代 {iteration}/{iterations}：采样 N={n_samples} 并打分")
        pairs: list[dict[str, Any]] = []
        base_model.eval()
        for idx, example in enumerate(local_examples):
            prompt_ids, completions, texts = sample_reasonings(
                base_model,
                tokenizer,
                example,
                n=n_samples,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
            )
            scores = score_fn(example, texts)
            if len(set(scores)) < 2:
                continue
            best = max(range(len(scores)), key=lambda i: scores[i])
            worst = min(range(len(scores)), key=lambda i: scores[i])
            pairs.append(
                {
                    "prompt_ids": prompt_ids,
                    "chosen": completions[best],
                    "rejected": completions[worst],
                    "score_gap": scores[best] - scores[worst],
                }
            )
            best_reasoning[str(example.get("user_id") or "")] = texts[best]
            if idx % 200 == 0 and idx:
                logger.info(f"[dpo]   采样 {idx}/{len(local_examples)}，已得 {len(pairs)} 对")

        if not pairs:
            logger.warning(f"[dpo] 迭代 {iteration} 没有构造出任何偏好对，提前结束")
            break
        logger.info(f"[dpo] 迭代 {iteration}：{len(pairs)} 个偏好对，开始训练")

        base_model.train()
        micro = 0
        optimizer.zero_grad()
        for epoch in range(epochs):
            for start in range(0, len(pairs), pair_batch_size):
                with timer.phase("batch_prep"):
                    batch = pairs[start : start + pair_batch_size]
                    prompts = [p["prompt_ids"] for p in batch]
                    chosens = [p["chosen"] for p in batch]
                    rejecteds = [p["rejected"] for p in batch]

                with runtime.autocast():
                    with timer.phase("policy_scoring"):
                        policy_chosen, policy_rejected = score_preference_batch(
                            train_model,
                            prompts,
                            chosens,
                            rejecteds,
                            pad_token_id=pad_id,
                            pad_to_multiple_of=pad_mult,
                        )
                    with timer.phase("reference_scoring"):
                        with torch.no_grad():
                            if ref_model is not None:
                                ref_chosen, ref_rejected = score_preference_batch(
                                    ref_model,
                                    prompts,
                                    chosens,
                                    rejecteds,
                                    pad_token_id=pad_id,
                                    pad_to_multiple_of=pad_mult,
                                )
                            else:
                                ref_chosen = policy_chosen.detach()
                                ref_rejected = policy_rejected.detach()
                    with timer.phase("loss"):
                        loss_raw, stats_acc = dpo_loss(
                            policy_chosen,
                            policy_rejected,
                            ref_chosen,
                            ref_rejected,
                            beta,
                        )
                        loss = loss_raw / accum

                with timer.phase("backward"):
                    if (micro + 1) % accum == 0:
                        runtime.backward(loss)
                    else:
                        with dist_utils.no_sync(train_model):
                            runtime.backward(loss)
                micro += 1

                if micro % accum:
                    continue

                with timer.phase("optimizer"):
                    runtime.optimizer_step(
                        optimizer,
                        parameters=[p for p in train_model.parameters() if p.requires_grad],
                        max_grad_norm=max_grad_norm,
                    )
                    optimizer.zero_grad()
                    scheduler.step()
                global_step += 1
                throughput["pairs_trained"] = throughput.get("pairs_trained", 0) + len(batch)

                should_log = global_step % logging_steps == 0
                if should_log:
                    packed = reduce_scalar_pack(
                        [
                            float(loss_raw.detach()),
                            stats_acc["dpo_margin"],
                            stats_acc["chosen_reward"],
                            stats_acc["rejected_reward"],
                            stats_acc["accuracy"],
                        ]
                    )
                    if is_main:
                        metrics = {
                            "loss": packed[0],
                            "iteration": iteration,
                            "dpo_margin": packed[1],
                            "chosen_reward": packed[2],
                            "rejected_reward": packed[3],
                            "accuracy": packed[4],
                        }
                        history.append({"step": global_step, **metrics})
                        logger.log_metrics(
                            metrics,
                            stage="dpo",
                            step=global_step,
                            epoch=float(epoch),
                            split="train",
                            wandb_prefix="train",
                        )

                for callback in callbacks:
                    hook = getattr(callback, "on_step", None)
                    if hook is not None:
                        hook(global_step, base_model)

                if should_save_at_step(global_step, save_every):
                    save_step_checkpoint(
                        train_model,
                        output_dir,
                        global_step,
                        tokenizer=tokenizer,
                        cfg=cfg,
                        logger=logger,
                        tag="dpo",
                    )

        if on_iteration_end is not None:
            on_iteration_end(iteration, dict(best_reasoning))

    for callback in callbacks:
        hook = getattr(callback, "on_train_end", None)
        if hook is not None:
            hook(global_step, base_model)

    dist_utils.barrier("dpo_end")
    final_dir = Path(output_dir) / "final"
    dist_utils.save_pretrained_distributed(train_model, final_dir, tokenizer=tokenizer, is_main=is_main)
    if is_main:
        (Path(output_dir) / "train_log.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
    dist_utils.barrier("dpo_saved")

    logger.info(f"[dpo] 完成 {global_step} steps，权重 → {final_dir}")
    phase = timer.summary()
    total_s = sum(v.get("total_s", 0.0) for v in phase.values()) if phase else 0.0
    perf: dict[str, Any] = {}
    pairs_n = int(throughput.get("pairs_trained") or 0)
    if total_s > 0 and pairs_n > 0:
        perf["pairs_per_sec"] = round(pairs_n / total_s, 3)
    vram = peak_vram_gb()
    if vram is not None:
        perf["peak_vram_gb"] = vram
    if phase:
        perf["phase_timers"] = phase
    if perf:
        cfg.setdefault("_performance", {})["dpo"] = perf
    return {
        "stage": "dpo",
        "checkpoint": str(final_dir),
        "steps": global_step,
        "iterations": iterations,
        "best_reasoning": best_reasoning,
        "mode": get_mode(cfg),
        "precision": runtime.precision.precision,
        "strategy": runtime.effective_strategy,
        "batch_plan": batch_plan.to_dict(),
        "performance": perf or None,
    }


__all__ = [
    "dpo_loss",
    "sample_reasonings",
    "score_preference_batch",
    "run_dpo",
    "sequence_logprobs",
]
