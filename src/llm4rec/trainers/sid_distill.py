"""SID distribution distillation (hard + soft MC + exposure KL).

Mapped from ``dragonfly90/llm4rec-bias`` ``src/llm4rec/sid_distill.py::main``.

Rewritten for Integrated: full-parameter training, existing checkpoint /
runtime / logger / distributed stack. No PEFT, no argparse entry, no MPS.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from llm4rec.core import distributed as dist_utils
from llm4rec.core.exceptions import ConfigurationError
from llm4rec.data.minionerec_distill import MiniOneRecDistillCollator, MiniOneRecDistillDataset
from llm4rec.sid.table import sid_token
from llm4rec.sid.transition import SidTransitionTeacher
from llm4rec.trainers.logprobs import sid_sequence_nll
from llm4rec.trainers.metrics_dist import reduce_scalar_pack


PROBABILITY_SUPPORTS = ("full_vocab", "valid_sid")


def level1_student_probs(
    first_logits: torch.Tensor,
    level1_ids: torch.Tensor,
    *,
    probability_support: str = "full_vocab",
) -> torch.Tensor:
    """Q_θ^(1) over first-level SID tokens.

    * ``full_vocab`` (default, reference-compatible): softmax over the full
      vocabulary, then gather level-1 SID token probabilities.
    * ``valid_sid``: softmax only over the legal level-1 SID tokens.
    """
    support = str(probability_support or "full_vocab")
    if support not in PROBABILITY_SUPPORTS:
        raise ConfigurationError(
            f"probability_support 必须是 {PROBABILITY_SUPPORTS}，得到 {support!r}"
        )
    if first_logits.numel() == 0:
        return first_logits.new_zeros((0, int(level1_ids.numel())))
    if support == "full_vocab":
        q_full = F.softmax(first_logits.float(), dim=-1)
        return q_full.index_select(-1, level1_ids)
    return F.softmax(first_logits.float().index_select(-1, level1_ids), dim=-1)


def _all_reduce_sum_keep_grad(tensor: torch.Tensor) -> torch.Tensor:
    """Differentiable all-reduce SUM so exposure KL uses the global batch mean."""
    if not dist_utils.is_distributed():
        return tensor
    try:
        from torch.distributed.nn.functional import all_reduce

        return all_reduce(tensor)
    except Exception:  # noqa: BLE001
        import torch.distributed as dist

        # Fallback: in-place all_reduce is not autograd-aware; still keep a
        # consistent statistic across ranks (grad will be local-only).
        out = tensor.contiguous()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out


def exposure_alignment_loss(
    student_q: torch.Tensor,
    teacher_p: torch.Tensor,
) -> torch.Tensor:
    """``KL(mean Q || mean P)`` with a cross-rank differentiable mean.

    Single-GPU and the all-reduced multi-GPU definition match: both are the
    size-weighted mean of the per-prompt level-1 distributions.
    """
    if student_q.numel() == 0:
        return student_q.new_zeros(())
    pack = torch.cat(
        [
            student_q.sum(0),
            teacher_p.sum(0),
            student_q.new_tensor([float(student_q.size(0))]),
        ]
    )
    pack = _all_reduce_sum_keep_grad(pack)
    k = int(student_q.size(-1))
    q_sum, p_sum, count = pack[:k], pack[k : 2 * k], pack[-1].clamp(min=1.0)
    q_bar = (q_sum / count).clamp(min=1e-9)
    p_bar = (p_sum / count).clamp(min=1e-9)
    q_bar = q_bar / q_bar.sum()
    p_bar = p_bar / p_bar.sum()
    return (q_bar * (q_bar.log() - p_bar.log())).sum()


def hard_sid_loss(nll: torch.Tensor, gold_mask: torch.Tensor) -> torch.Tensor:
    n_prompts = gold_mask.float().sum().clamp(min=1.0)
    return (nll * gold_mask.float()).sum() / n_prompts


def soft_sid_loss(nll: torch.Tensor, soft_weight: torch.Tensor, n_prompts: int | torch.Tensor) -> torch.Tensor:
    denom = n_prompts if torch.is_tensor(n_prompts) else nll.new_tensor(float(n_prompts))
    return (nll * soft_weight).sum() / denom.clamp(min=1.0)


def distill_total_loss(
    hard: torch.Tensor,
    soft: torch.Tensor,
    exposure: torch.Tensor,
    *,
    hard_weight: float,
    exposure_weight: float,
) -> torch.Tensor:
    alpha = float(hard_weight)
    return alpha * hard + (1.0 - alpha) * soft + float(exposure_weight) * exposure


def _level1_token_ids(tokenizer: Any, sid_table: Any) -> list[int]:
    k1 = int(sid_table.level_codebook_sizes()[0])
    ids = [
        int(tokenizer.convert_tokens_to_ids(sid_token(0, code, sid_table.prefixes)))
        for code in range(k1)
    ]
    if any(i < 0 for i in ids):
        raise ConfigurationError("第一层 SID token 不在 tokenizer 词表里")
    return ids


def _move_sequences(
    prompt_ids: Sequence[torch.Tensor],
    completion_ids: Sequence[torch.Tensor],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    return (
        [p.to(device) for p in prompt_ids],
        [c.to(device) for c in completion_ids],
    )


def _resolve_transition_path(cfg: dict[str, Any], artifacts: dict[str, Any] | None) -> str:
    arts = artifacts or {}
    if arts.get("transition_checkpoint"):
        return str(arts["transition_checkpoint"])
    distill_cfg = ((cfg.get("train") or {}).get("distill") or {})
    if distill_cfg.get("transition_checkpoint"):
        return str(distill_cfg["transition_checkpoint"])
    trans_cfg = ((cfg.get("train") or {}).get("transition") or {})
    if trans_cfg.get("checkpoint"):
        return str(trans_cfg["checkpoint"])
    raise ConfigurationError(
        "distill 需要 Transition 教师：先跑 transition stage，"
        "或设置 train.distill.transition_checkpoint"
    )


def run_sid_distill(
    *,
    cfg: dict[str, Any],
    model: Any,
    tokenizer: Any,
    sid_table: Any,
    catalog: Any | None,
    train_examples: Sequence[dict[str, Any]],
    eval_examples: Sequence[dict[str, Any]] | None,
    output_dir: Path,
    logger: Any,
    runtime: Any = None,
    artifacts: dict[str, Any] | None = None,
    callbacks: Sequence[Any] = (),
) -> dict[str, Any]:
    """Distill Transition soft targets back into the MiniOneRec LLM."""
    dcfg = dict((cfg.get("train") or {}).get("distill") or {})
    if not train_examples:
        raise ConfigurationError("distill 训练集为空")

    samples = int(dcfg.get("samples_per_prompt") or dcfg.get("top_m") or 8)
    hard_weight = float(dcfg.get("hard_weight") if dcfg.get("hard_weight") is not None else 0.5)
    exposure_weight = float(dcfg.get("exposure_weight") or 0.0)
    support = str(dcfg.get("probability_support") or "full_vocab")
    if support not in PROBABILITY_SUPPORTS:
        raise ConfigurationError(
            f"probability_support 必须是 {PROBABILITY_SUPPORTS}，得到 {support!r}"
        )

    if runtime is None:
        from llm4rec.runtime.context import build_runtime

        runtime = build_runtime(cfg, log=logger.info)
    runtime.bind_model_params(model, stage="distill", log=logger.info)

    if torch.cuda.is_available():
        expected = torch.device(f"cuda:{dist_utils.local_rank()}")
        cur = next(model.parameters()).device
        if cur != expected:
            model = model.to(expected)

    device = next(model.parameters()).device
    teacher_path = _resolve_transition_path(cfg, artifacts)
    trans_cfg = ((cfg.get("train") or {}).get("transition") or {})
    teacher = SidTransitionTeacher.from_checkpoint(
        teacher_path,
        sid_table,
        catalog=catalog,
        device=device,
        temperature=dcfg.get("temperature", trans_cfg.get("temperature")),
        target_smoothing=dcfg.get("target_smoothing", trans_cfg.get("target_smoothing")),
        popularity_gamma=dcfg.get("popularity_gamma", trans_cfg.get("popularity_gamma")),
    )
    logger.info(f"[distill] teacher ← {teacher_path}  catalog={len(teacher.items)}")

    max_len = int(dcfg.get("max_seq_length") or ((cfg.get("train") or {}).get("sft") or {}).get("max_seq_length") or 1024)
    seed = int(cfg.get("seed") or 42)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + dist_utils.rank())
    train_ds = MiniOneRecDistillDataset(train_examples, sid_table)
    eval_ds = MiniOneRecDistillDataset(eval_examples or [], sid_table) if eval_examples else None
    n_train_global = len(train_ds)
    n_eval_global = len(eval_ds) if eval_ds is not None else 0
    if n_train_global == 0:
        raise ConfigurationError("distill 过滤后训练集为空")

    if dist_utils.is_distributed():
        from torch.utils.data import Subset

        train_ds = Subset(train_ds, dist_utils.shard(list(range(n_train_global))))
        if eval_ds is not None and len(eval_ds):
            eval_ds = Subset(eval_ds, dist_utils.shard(list(range(len(eval_ds)))))

    if not len(train_ds):
        raise ConfigurationError(
            f"rank {dist_utils.rank()} 分到 0 条 distill 样本 —— 训练集比卡数还少"
        )

    collator = MiniOneRecDistillCollator(
        tokenizer,
        sid_table,
        teacher,
        samples_per_prompt=samples,
        max_length=max_len,
        catalog_chunk_size=int(dcfg.get("catalog_chunk_size") or 256),
        generator=gen,
    )
    preferred = int(dcfg.get("preferred_per_device_batch_size") or dcfg.get("per_device_batch_size") or 4)
    dcfg.setdefault("per_device_batch_size", preferred)
    batch_plan = runtime.resolve_stage_batch("distill", dcfg)
    for line in batch_plan.summary_lines():
        logger.info(f"[distill] {line}")

    per_device_b = int(batch_plan.per_device_batch_size)
    accum = int(batch_plan.gradient_accumulation_steps)
    max_steps_cfg = dcfg.get("max_steps")
    if max_steps_cfg in (None, 0, "null"):
        epochs = float(dcfg.get("epochs") or 1)
        steps_per_epoch = max(1, math.ceil(len(train_ds) / max(per_device_b, 1) / max(accum, 1)))
        max_steps = int(math.ceil(steps_per_epoch * epochs))
    else:
        max_steps = int(max_steps_cfg)
        epochs = float(dcfg.get("epochs") or 1)

    from llm4rec.runtime.activation_ckpt import resolve_activation_checkpointing
    from llm4rec.runtime.checkpointing import (
        is_better_metric,
        resolve_save_steps,
        save_best_checkpoint,
        save_best_enabled,
        save_step_checkpoint,
        should_save_at_step,
    )
    from llm4rec.trainers.schedulers import build_optimizer, create_scheduler

    hw_cfg = cfg.get("hardware") or {}
    grad_ckpt, act_reason = resolve_activation_checkpointing(
        hw_cfg,
        preferred_micro=preferred,
        selected_micro=per_device_b,
        pressure_ratio=getattr(runtime.strategy, "pressure_ratio", None),
        effective_strategy=runtime.effective_strategy or runtime.strategy.strategy,
        strategy_source=getattr(runtime.strategy, "source", None),
    )
    if grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
        logger.info(f"[distill] gradient checkpointing on ({act_reason})")

    strategy = runtime.effective_strategy or runtime.strategy.strategy
    if dist_utils.is_main():
        logger.info("[distill] skip torch.compile（变长 batch + SID vocab，inductor 不稳定）")
    train_core = model
    if strategy == "fsdp" and runtime.world_size > 1:
        train_model = runtime.wrap_model(train_core)
    else:
        train_model = train_core
        if dist_utils.is_main() and dist_utils.is_distributed():
            logger.info("[distill] skip DDP wrap; shard + manual grad allreduce")

    optimizer, optim_fallback = build_optimizer(
        [p for p in train_model.parameters() if p.requires_grad],
        lr=float(dcfg.get("learning_rate") or 1e-5),
        weight_decay=float(dcfg.get("weight_decay") or 0.0),
        optim_name=dcfg.get("optim") or dcfg.get("optimizer"),
    )
    if optim_fallback and dist_utils.is_main():
        logger.info(f"[distill] optimizer fallback: {optim_fallback}")
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=str(dcfg.get("lr_scheduler_type") or "cosine"),
        num_training_steps=max_steps,
        warmup_ratio=dcfg.get("warmup_ratio"),
        warmup_steps=dcfg.get("warmup_steps"),
    )
    max_grad_norm = float(dcfg.get("max_grad_norm") or 1.0)
    logging_steps = max(1, int(dcfg.get("logging_steps") or 10))
    eval_steps = dcfg.get("eval_steps")
    eval_every = (
        int(eval_steps)
        if eval_ds is not None and eval_steps not in (None, 0, "null", False)
        else None
    )
    save_every = resolve_save_steps(cfg, dcfg, max_steps=max_steps, as_int=True)
    level1_ids = torch.tensor(_level1_token_ids(tokenizer, sid_table), device=device, dtype=torch.long)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    pad_mult = ((cfg.get("optimization") or {}).get("generation") or {}).get("pad_to_multiple_of")
    pad_mult_i = int(pad_mult) if pad_mult not in (None, 0, "null", False) else None

    order = list(range(len(train_ds)))
    cursor = 0
    step = 0
    last = {"loss": 0.0, "hard": 0.0, "soft": 0.0, "exposure": 0.0}
    last_eval = None
    best_eval: float | None = None
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    samples_seen = 0
    is_main = dist_utils.is_main()
    optimizer.zero_grad(set_to_none=True)
    train_model.train()
    logger.info(
        f"[distill] prompts={n_train_global} (rank {len(train_ds)}) "
        f"samples_per_prompt={samples} hard_weight={hard_weight} "
        f"exposure_weight={exposure_weight} support={support} "
        f"{dist_utils.summary_line()}"
    )

    from llm4rec.tracking.progress import overwrite_progress

    progress_interval = float(dcfg.get("progress_log_interval_s") or 5.0)
    with overwrite_progress(
        max_steps,
        "distill",
        global_total=max_steps,
        name="distill",
        log_interval_s=progress_interval,
    ) as progress:
        while step < max_steps:
            micro = {"loss": 0.0, "hard": 0.0, "soft": 0.0, "exposure": 0.0}
            for _ in range(accum):
                idxs = []
                for _ in range(per_device_b):
                    idxs.append(order[cursor % len(order)])
                    cursor += 1
                rows = [train_ds[int(i)] for i in idxs]
                batch = collator(rows)
                prompts, comps = _move_sequences(
                    batch["prompt_ids"], batch["completion_ids"], device
                )
                soft_w = batch["soft_weight"].to(device)
                gold = batch["gold_mask"].to(device)
                teacher_p = batch["level1_teacher"].to(device)
                with runtime.autocast():
                    nll, first_logits = sid_sequence_nll(
                        train_model,
                        prompts,
                        comps,
                        pad_token_id=pad_id,
                        pad_to_multiple_of=pad_mult_i,
                    )
                    hard = hard_sid_loss(nll, gold)
                    soft = soft_sid_loss(nll, soft_w, int(batch["n_prompts"]))
                    if exposure_weight:
                        q = level1_student_probs(
                            first_logits[gold],
                            level1_ids,
                            probability_support=support,
                        )
                        exposure = exposure_alignment_loss(q, teacher_p)
                    else:
                        exposure = nll.new_zeros(())
                    loss = distill_total_loss(
                        hard,
                        soft,
                        exposure,
                        hard_weight=hard_weight,
                        exposure_weight=exposure_weight,
                    )
                    loss_b = loss / accum
                runtime.backward(loss_b)
                micro["loss"] += float(loss.detach())
                micro["hard"] += float(hard.detach())
                micro["soft"] += float(soft.detach())
                micro["exposure"] += float(exposure.detach())
                samples_seen += int(batch["n_prompts"])
                del nll, first_logits, loss, loss_b, hard, soft, exposure, batch

            if not dist_utils.is_fsdp(train_model):
                dist_utils.allreduce_gradients(train_model)
            runtime.optimizer_step(
                optimizer,
                parameters=[p for p in train_model.parameters() if p.requires_grad],
                max_grad_norm=max_grad_norm,
            )
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            last = {k: v / max(accum, 1) for k, v in micro.items()}
            if is_main and progress.bar is not None:
                progress.bar.set_postfix(
                    loss=f"{last['loss']:.4f}",
                    hard=f"{last['hard']:.4f}",
                    soft=f"{last['soft']:.4f}",
                    refresh=False,
                )
            progress.update(1)

            should_log = step % logging_steps == 0 or step == max_steps
            if should_log:
                packed = reduce_scalar_pack(
                    [last["loss"], last["hard"], last["soft"], last["exposure"]]
                )
                if is_main:
                    metrics = {
                        "loss": packed[0],
                        "hard": packed[1],
                        "soft": packed[2],
                        "exposure": packed[3],
                        "lr": float(scheduler.get_last_lr()[0]),
                    }
                    history.append({"step": step, **metrics})
                    logger.log_metrics(
                        metrics,
                        stage="distill",
                        step=step,
                        epoch=epochs * step / max(max_steps, 1),
                        split="train",
                        wandb_prefix="train",
                    )
                    if progress.bar is None:
                        logger.info(
                            f"[distill] step={step}/{max_steps} loss={metrics['loss']:.4f} "
                            f"hard={metrics['hard']:.4f} soft={metrics['soft']:.4f} "
                            f"exp_kl={metrics['exposure']:.4f}"
                        )
            for cb in callbacks:
                if hasattr(cb, "on_step"):
                    cb.on_step(step, train_model)

            if eval_every is not None and step % eval_every == 0:
                last_eval = _eval_hard_nll(
                    train_model,
                    eval_ds,
                    collator=collator,
                    device=device,
                    runtime=runtime,
                    pad_id=pad_id,
                    pad_mult=pad_mult_i,
                    per_device_batch=per_device_b,
                )
                train_model.train()
                if last_eval is not None and is_main:
                    logger.log_metrics(
                        {"hard_nll": last_eval},
                        stage="distill",
                        step=step,
                        split="eval",
                        wandb_prefix="eval",
                    )
                    logger.info(f"[distill] eval step={step} hard_nll={last_eval:.4f}")
                if (
                    last_eval is not None
                    and save_best_enabled(cfg)
                    and is_better_metric(last_eval, best_eval)
                ):
                    best_eval = last_eval
                    save_best_checkpoint(
                        train_model,
                        output_dir,
                        metric=last_eval,
                        step=step,
                        tokenizer=tokenizer,
                        logger=logger,
                        tag="distill",
                        metric_name="eval_hard_nll",
                    )

            if should_save_at_step(step, save_every):
                save_step_checkpoint(
                    train_model,
                    output_dir,
                    step,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    logger=logger,
                    tag="distill",
                )

    for cb in callbacks:
        if hasattr(cb, "on_train_end"):
            cb.on_train_end(step, train_model)

    dist_utils.barrier("distill_end")
    final_dir = Path(output_dir) / "final"
    dist_utils.save_pretrained_distributed(
        train_model, final_dir, tokenizer=tokenizer, is_main=is_main
    )
    if is_main:
        (Path(output_dir) / "train_log.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
    dist_utils.barrier("distill_saved")

    elapsed = max(1e-6, time.perf_counter() - t0)
    perf = {
        "samples_per_sec": round(samples_seen * max(dist_utils.world_size(), 1) / elapsed, 3),
        "optimizer_steps_per_sec": round(step / elapsed, 3),
    }
    cfg.setdefault("_performance", {})["distill"] = perf
    metrics = {"train_loss": last["loss"], "train_steps": step, **last}
    if last_eval is not None:
        metrics["eval_hard_nll"] = last_eval
    if best_eval is not None:
        metrics["best_eval_hard_nll"] = best_eval
    logger.info(f"[distill] 完成，权重 → {final_dir}")
    return {
        "stage": "distill",
        "checkpoint": str(final_dir),
        "metrics": metrics,
        "n_train": n_train_global,
        "n_eval": n_eval_global,
        "batch_plan": batch_plan.to_dict(),
        "performance": perf,
        "teacher": str(teacher_path),
        "probability_support": support,
    }


@torch.no_grad()
def _eval_hard_nll(
    model: Any,
    eval_ds: Any,
    *,
    collator: MiniOneRecDistillCollator,
    device: torch.device,
    runtime: Any,
    pad_id: int,
    pad_mult: int | None,
    per_device_batch: int,
) -> float | None:
    if eval_ds is None or len(eval_ds) == 0:
        return None
    model.eval()
    total = 0.0
    count = 0
    bs = max(1, int(per_device_batch))
    for start in range(0, len(eval_ds), bs):
        idxs = list(range(start, min(start + bs, len(eval_ds))))
        rows = [eval_ds[int(i)] for i in idxs]
        batch = collator(rows)
        prompts, comps = _move_sequences(batch["prompt_ids"], batch["completion_ids"], device)
        gold = batch["gold_mask"].to(device)
        with runtime.autocast():
            nll, _ = sid_sequence_nll(
                model, prompts, comps, pad_token_id=pad_id, pad_to_multiple_of=pad_mult
            )
            hard = hard_sid_loss(nll, gold)
        total += float(hard.detach()) * int(gold.sum().item())
        count += int(gold.sum().item())
        del nll, batch
    packed = reduce_scalar_pack([total, float(count)], op="sum")
    if packed[1] <= 0:
        return None
    return packed[0] / packed[1]
