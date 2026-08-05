"""DPO4Rec 的 DPO 训练 —— 对齐 arXiv 2410.05939 的 Algorithm 1。

    for iteration in 1..T:                       # 论文实测 T=2 最好，T=3 因过拟合退化
        for each prompt:
            采 N 份推理文本                       # 论文 N=10
            用 reranker 给每份打分（NDCG@5）      # reranker 就是 reward model
            chosen   = argmax(score)
            rejected = argmin(score)
        用 (prompt, chosen, rejected) 训 DPO      # beta=0.01

DPO loss（论文式 (1)）：
    L = -log σ( β·log[π(y_w|x)/π_ref(y_w|x)] − β·log[π(y_l|x)/π_ref(y_l|x)] )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

from llm4rec.core import distributed as dist_utils
from llm4rec.core.exceptions import ConfigurationError
from llm4rec.trainers.grpo import sequence_logprobs


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """标准 sigmoid DPO。返回 ``(loss, 诊断指标)``。"""
    chosen_logratio = policy_chosen - ref_chosen
    rejected_logratio = policy_rejected - ref_rejected
    margin = beta * (chosen_logratio - rejected_logratio)
    loss = -F.logsigmoid(margin).mean()
    stats = {
        "dpo_margin": float(margin.mean()),
        "chosen_reward": float(beta * chosen_logratio.mean()),
        "rejected_reward": float(beta * rejected_logratio.mean()),
        # 偏好方向对了的比例 —— DPO 最重要的健康指标
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
) -> tuple[torch.Tensor, list[torch.Tensor], list[str]]:
    """对一个 prompt 采 N 份推理文本（论文 N=10）。"""
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
) -> dict[str, Any]:
    """DPO4Rec 主循环（含论文的迭代优化）。"""
    dpo_cfg = (cfg.get("train") or {}).get("dpo") or {}
    if not train_examples:
        raise ConfigurationError("DPO 训练集为空")

    iterations = int(dpo_cfg.get("iterations") or 2)
    n_samples = int(dpo_cfg.get("num_samples") or 10)
    beta = float(dpo_cfg.get("beta") or 0.01)
    temperature = float(dpo_cfg.get("sampling_temperature") or 1.0)
    max_new_tokens = int(dpo_cfg.get("max_new_tokens") or 512)
    epochs = int(dpo_cfg.get("epochs") or 3)
    accum = int(dpo_cfg.get("gradient_accumulation_steps") or 8)
    logging_steps = int(dpo_cfg.get("logging_steps") or 1)

    # 多卡：各 rank 采不同样本，梯度由 DDP 在 backward 里 all-reduce
    local_examples = dist_utils.shard(train_examples)
    if not local_examples:
        raise ConfigurationError(
            f"rank {dist_utils.rank()} 分到 0 条样本 —— 训练集比卡数还少"
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(dpo_cfg.get("learning_rate") or 5e-5),
        betas=(float(dpo_cfg.get("adam_beta1") or 0.9), float(dpo_cfg.get("adam_beta2") or 0.999)),
        eps=float(dpo_cfg.get("adam_epsilon") or 1e-8),
        weight_decay=float(dpo_cfg.get("weight_decay") or 0.0),
    )
    max_grad_norm = float(dpo_cfg.get("max_grad_norm") or 1.0)

    base_model = model                       # 采样 / 存盘（generate 不能走 DDP）
    train_model = dist_utils.wrap_ddp(model)  # 反向传播
    is_main = dist_utils.is_main()

    if ref_model is not None:
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    logger.info(
        f"[dpo] 开始：iterations={iterations} N={n_samples} beta={beta} "
        f"epochs={epochs} n_train={len(train_examples)} "
        f"(本 rank {len(local_examples)})  {dist_utils.summary_line()}"
    )

    global_step = 0
    history: list[dict[str, Any]] = []
    best_reasoning: dict[str, str] = {}

    for iteration in range(1, iterations + 1):
        # ---------- 1) 采样 + 打分 → 构造偏好对 ----------
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
            )
            scores = score_fn(example, texts)
            if len(set(scores)) < 2:
                # N 份全同分 → 没有偏好信号，跳过（论文的 sample filter）
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

        # ---------- 2) DPO 训练 ----------
        base_model.train()
        micro = 0
        optimizer.zero_grad()
        for epoch in range(epochs):
            for pair in pairs:
                policy_chosen = sequence_logprobs(train_model, pair["prompt_ids"], pair["chosen"]).sum()
                policy_rejected = sequence_logprobs(train_model, pair["prompt_ids"], pair["rejected"]).sum()
                with torch.no_grad():
                    if ref_model is not None:
                        ref_chosen = sequence_logprobs(ref_model, pair["prompt_ids"], pair["chosen"]).sum()
                        ref_rejected = sequence_logprobs(ref_model, pair["prompt_ids"], pair["rejected"]).sum()
                    else:
                        ref_chosen = policy_chosen.detach()
                        ref_rejected = policy_rejected.detach()

                loss, stats = dpo_loss(
                    policy_chosen.unsqueeze(0),
                    policy_rejected.unsqueeze(0),
                    ref_chosen.unsqueeze(0),
                    ref_rejected.unsqueeze(0),
                    beta,
                )
                # 累积中间步跳过 all-reduce，只在最后一个 micro-step 同步
                if (micro + 1) % accum == 0:
                    (loss / accum).backward()
                else:
                    with dist_utils.no_sync(train_model):
                        (loss / accum).backward()
                micro += 1

                if micro % accum:
                    continue
                torch.nn.utils.clip_grad_norm_(
                    [p for p in base_model.parameters() if p.requires_grad], max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % logging_steps == 0 and is_main:
                    metrics = {
                        "loss": dist_utils.all_reduce_mean(float(loss.detach())),
                        "iteration": iteration,
                        **stats,
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

        # ---------- 3) 迭代收尾：把最好的推理文本回灌给 reranker ----------
        # 论文 §IV-C-2 的"双向增益"：LLM 变好 → reranker 也能变好
        if on_iteration_end is not None:
            on_iteration_end(iteration, dict(best_reasoning))

    for callback in callbacks:
        hook = getattr(callback, "on_train_end", None)
        if hook is not None:
            hook(global_step, base_model)

    dist_utils.barrier("dpo_end")
    final_dir = Path(output_dir) / "final"
    if is_main:
        final_dir.mkdir(parents=True, exist_ok=True)
        dist_utils.unwrap(base_model).save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        (Path(output_dir) / "train_log.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
    dist_utils.barrier("dpo_saved")

    logger.info(f"[dpo] 完成 {global_step} steps，权重 → {final_dir}")
    return {
        "stage": "dpo",
        "checkpoint": str(final_dir),
        "steps": global_step,
        "iterations": iterations,
        "best_reasoning": best_reasoning,
    }
