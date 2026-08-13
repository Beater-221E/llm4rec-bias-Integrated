"""全参 SFT —— 三条路线共用。

★ 没有 LoRA。官方 MiniOneRec 的 ``sft.py`` 里连 ``import peft`` 都没有，
  只有一个 ``freeze_LLM`` 开关（默认 False = 全参）。而且我们后面要做
  表征分析，LoRA 会低估 RL 对表征的改动。

三条路线的差别只在样本怎么构建（``llm4rec.data.examples``），
训练循环、checkpoint 格式、日志全部共用。

SFT 阶段【不做】bias 在线评测 —— 按研究设计，SFT 只提供一个基线，
bias 的漂移观测放在 RL/DPO 阶段。stage 结束后会评一次基线。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from llm4rec.core.exceptions import ConfigurationError

IGNORE_INDEX = -100


def _torch_compile_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map optimization.compile → HF TrainingArguments when supported.

    ``auto`` skips compile on pre-Ampere GPUs (cc < 8): inductor first-step
    compile is very slow there and HF itself warns speedups need Ampere+.
    """
    compile_cfg = (cfg.get("optimization") or {}).get("compile") or {}
    enabled = compile_cfg.get("enabled", "auto")
    mode = str(cfg.get("mode") or (cfg.get("experiment") or {}).get("mode") or "integrated")
    if enabled in (False, "false", "False", None, "null"):
        return {}
    if enabled == "auto" and mode == "reproduction":
        return {}
    if enabled == "auto" and torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability(0)
        if major < 8:
            return {}
    if enabled not in (True, "true", "True", "auto"):
        return {}
    out: dict[str, Any] = {"torch_compile": True}
    backend = compile_cfg.get("backend")
    if backend:
        out["torch_compile_backend"] = str(backend)
    cmode = compile_cfg.get("mode")
    if cmode:
        out["torch_compile_mode"] = str(cmode)
    return out


def _length_group_kwargs(sft_cfg: dict[str, Any]) -> dict[str, Any]:
    """HF 5.x renamed ``group_by_length`` → ``train_sampling_strategy``."""
    import inspect

    enabled = bool(sft_cfg.get("group_by_length", False))
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "train_sampling_strategy" in params:
        return {
            "train_sampling_strategy": "group_by_length" if enabled else "random",
        }
    if "group_by_length" in params:
        return {"group_by_length": enabled}
    return {}


def _disable_tqdm(sft_cfg: dict[str, Any]) -> bool:
    """Overwrite-style tqdm only works on a real TTY.

    ``run.sh`` pipes through ``tee``, so stderr is a pipe: tqdm falls back to
    printing a *new line every step* and blows up the log. In that case disable
    the bar and keep periodic ``logging_steps`` metrics only.
    """
    import sys

    if "disable_tqdm" in sft_cfg and sft_cfg.get("disable_tqdm") is not None:
        return bool(sft_cfg.get("disable_tqdm"))
    return not (sys.stderr.isatty() and sys.stdout.isatty())


# ------------------------------------------------------------------ Dataset


class ChatSFTDataset(Dataset):
    """把 ``{prompt: messages, answer: str}`` 编成带 label mask 的样本。

    只有 answer 部分算 loss（prompt 全部 mask 成 -100）—— 这是 SFT 的基本要求，
    不 mask 的话模型会去拟合 prompt 里的历史，等于变相泄漏。
    """

    def __init__(
        self,
        examples: Sequence[dict[str, Any]],
        tokenizer: Any,
        max_length: int = 1024,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        tok = self.tokenizer

        prompt_ids = tok.apply_chat_template(
            ex["prompt"], add_generation_prompt=True, tokenize=True
        )
        # 兼容偶发的 BatchEncoding / str 返回
        if hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids["input_ids"]
        if isinstance(prompt_ids, str):
            prompt_ids = tok(prompt_ids, add_special_tokens=False)["input_ids"]
        answer_ids = tok(str(ex["answer"]), add_special_tokens=False)["input_ids"]
        eos = tok.eos_token_id
        if eos is not None:
            answer_ids = answer_ids + [eos]

        input_ids = [int(t) for t in prompt_ids] + [int(t) for t in answer_ids]
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(answer_ids)

        # 超长时从【左边】截断：保住答案，砍掉最早的历史
        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }


@dataclass
class PadCollator:
    """Padding; label 用 -100 填充。Optional pad_to_multiple_of + padding_side."""

    pad_token_id: int
    pad_to_multiple_of: int | None = None
    padding_side: str = "right"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of and self.pad_to_multiple_of > 1:
            rem = max_len % self.pad_to_multiple_of
            if rem:
                max_len += self.pad_to_multiple_of - rem
        left = str(self.padding_side).lower() == "left"
        batch: dict[str, list[list[int]]] = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            if left:
                batch["input_ids"].append([self.pad_token_id] * pad + f["input_ids"])
                batch["labels"].append([IGNORE_INDEX] * pad + f["labels"])
                batch["attention_mask"].append([0] * pad + f["attention_mask"])
            else:
                batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
                batch["labels"].append(f["labels"] + [IGNORE_INDEX] * pad)
                batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


# ------------------------------------------------------------------ 日志回调


class MetricsCallback(TrainerCallback):
    """把 HF Trainer 的 log 转发到我们的 logger（jsonl + wandb）。"""

    def __init__(self, logger: Any, stage: str) -> None:
        self.logger = logger
        self.stage = stage

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:  # noqa: ANN001
        if not logs:
            return
        scalars = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        if not scalars:
            return
        split = "eval" if any(k.startswith("eval_") for k in scalars) else "train"
        self.logger.log_metrics(
            scalars,
            stage=self.stage,
            step=int(state.global_step),
            epoch=float(state.epoch) if state.epoch is not None else None,
            split=split,
            wandb_prefix="train" if split == "train" else "eval",
        )


class ThroughputCallback(TrainerCallback):
    """Coarse samples/sec + tokens/sec + peak VRAM for SFT (windowed)."""

    def __init__(self, cfg: dict[str, Any], *, window: int = 20) -> None:
        import time

        self.cfg = cfg
        self.window = max(1, int(window))
        self._time = time
        self._t0: float | None = None
        self._last_step = 0
        self._samples = 0
        self._tokens = 0
        self.metrics: dict[str, float] = {}

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        self._t0 = self._time.perf_counter()
        self._last_step = 0
        self._samples = 0
        self._tokens = 0

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        # Approximate: microbatch × world × accum counted as samples per optimizer step
        bs = int(args.per_device_train_batch_size or 1)
        ws = max(1, int(getattr(args, "world_size", 1) or 1))
        accum = max(1, int(args.gradient_accumulation_steps or 1))
        self._samples += bs * ws * accum
        # tokens unknown without batch; leave tokens/sec if trainer logs them
        if state.global_step > 0 and state.global_step % self.window == 0 and self._t0:
            dt = max(1e-6, self._time.perf_counter() - self._t0)
            steps = max(1, state.global_step - self._last_step)
            self.metrics = {
                "samples_per_sec": round(self._samples / dt, 3),
                "optimizer_steps_per_sec": round(steps / dt, 3),
            }
            from llm4rec.runtime.profiler import peak_vram_gb

            vram = peak_vram_gb()
            if vram is not None:
                self.metrics["peak_vram_gb"] = vram
            self._t0 = self._time.perf_counter()
            self._last_step = state.global_step
            self._samples = 0

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001
        if self.metrics:
            self.cfg.setdefault("_performance", {})["sft"] = dict(self.metrics)



# ------------------------------------------------------------------ 训练入口


def _resolve_deepspeed(cfg: dict[str, Any], logger: Any) -> dict[str, Any] | None:
    """``hardware.deepspeed`` → DeepSpeed 配置 dict。

    配置本身也是 YAML（``configs/deepspeed/<name>.yaml``），和其它配置层一样
    走 compose 加载 —— 不再有游离在配置体系外的 json。
    HF ``TrainingArguments`` 直接接受 dict，所以不用落临时文件。
    """
    name = (cfg.get("hardware") or {}).get("deepspeed")
    if not name or str(name).lower() in ("null", "none", "false"):
        return None

    # 已经在 compose 阶段被合并进来了（deepspeed_config 顶层键）
    inline = cfg.get("deepspeed_config")
    if isinstance(inline, dict) and inline:
        ds_config = inline
    else:
        from llm4rec.core.compose import load_layer

        layer = load_layer(f"deepspeed/{name}")
        ds_config = layer.get("deepspeed_config")
        if not isinstance(ds_config, dict):
            raise ConfigurationError(
                f"configs/deepspeed/{name}.yaml 里缺少 deepspeed_config 段"
            )

    if "zero3" in str(name):
        logger.warning(
            "[sft] 用了 ZeRO-3：参数被切分，RL 阶段的 generate 每步都要 gather 参数。"
            "本框架的 RL 走 DDP 不受影响，但要确认单卡放得下模型。"
        )
    stage = (ds_config.get("zero_optimization") or {}).get("stage", "?")
    offload = (
        (ds_config.get("zero_optimization") or {}).get("offload_optimizer") or {}
    ).get("device", "none")
    logger.info(f"[sft] DeepSpeed 已启用：{name}（ZeRO-{stage}，optimizer offload={offload}）")
    return ds_config


def run_sft(
    *,
    cfg: dict[str, Any],
    model: Any,
    tokenizer: Any,
    train_examples: Sequence[dict[str, Any]],
    eval_examples: Sequence[dict[str, Any]],
    output_dir: Path,
    logger: Any,
    stage: str = "sft",
    runtime: Any = None,
) -> dict[str, Any]:
    """跑全参 SFT，把完整权重存到 ``output_dir/final``。"""
    sft_cfg = (cfg.get("train") or {}).get("sft") or {}
    if not train_examples:
        raise ConfigurationError("SFT 训练集为空")

    max_len = int(sft_cfg.get("max_seq_length") or 1024)
    mode = str(cfg.get("mode") or (cfg.get("experiment") or {}).get("mode") or "integrated")
    route = str((cfg.get("experiment") or {}).get("route") or "")
    reference_sft = mode == "reproduction" and route == "minionerec"

    if reference_sft:
        from llm4rec.data.minionerec_sft import (
            MiniOneRecReferenceSFTDataset,
            sft_dataset_counts,
        )

        train_ds = MiniOneRecReferenceSFTDataset(train_examples, tokenizer, max_len)
        eval_ds = (
            MiniOneRecReferenceSFTDataset(eval_examples, tokenizer, max_len)
            if eval_examples
            else None
        )
        counts = sft_dataset_counts(train_examples)
        logger.info(f"[sft] MiniOneRec reproduction SFT counts: {counts}")
        padding_side = "left"
    else:
        train_ds = ChatSFTDataset(train_examples, tokenizer, max_len)
        eval_ds = ChatSFTDataset(eval_examples, tokenizer, max_len) if eval_examples else None
        padding_side = "right"

    if runtime is None:
        from llm4rec.runtime.context import build_runtime

        runtime = build_runtime(cfg, log=logger.info)
    runtime.bind_model_params(model, stage="sft", log=logger.info)

    # Ensure each torchrun rank owns its LOCAL_RANK device before probe/train.
    from llm4rec.core import distributed as dist_utils

    if torch.cuda.is_available():
        expected = torch.device(f"cuda:{dist_utils.local_rank()}")
        cur = next(model.parameters()).device
        if cur != expected:
            logger.warning(f"[sft] model on {cur}, moving to {expected}")
            model = model.to(expected)

    precision = runtime.precision.precision
    cfg["hardware"]["precision"] = precision
    preferred = int(
        sft_cfg.get("preferred_per_device_batch_size")
        or sft_cfg.get("per_device_batch_size")
        or 2
    )
    hw_cfg = cfg.get("hardware") or {}
    memory_auto = str(hw_cfg.get("memory") or "").lower() == "auto"
    pad_mult = ((cfg.get("optimization") or {}).get("generation") or {}).get("pad_to_multiple_of")
    pad_mult_i = int(pad_mult) if pad_mult not in (None, 0, "null", False) else None
    collator = PadCollator(
        tokenizer.pad_token_id or tokenizer.eos_token_id or 0,
        pad_to_multiple_of=pad_mult_i,
        padding_side=padding_side,
    )

    if memory_auto and len(train_ds) > 0:
        from llm4rec.runtime.memory import auto_tune_micro_batch
        from llm4rec.runtime.memory_estimate import select_memory_probe_examples

        # Sync before probe; actual OOM probe runs on rank0 only (see memory.py).
        dist_utils.barrier("sft-before-memory-probe")
        device = next(model.parameters()).device
        seed = int(cfg.get("seed") or 42)

        def probe(micro_b: int) -> None:
            # Length from raw examples (no chat-template tokenize for scanning).
            def _rough_len(row: Any) -> int:
                if not isinstance(row, dict):
                    return 0
                prompt = row.get("prompt") or []
                answer = str(row.get("answer") or "")
                text_chars = sum(
                    len(str(m.get("content") or ""))
                    for m in prompt
                    if isinstance(m, dict)
                )
                return text_chars + len(answer)

            idxs = select_memory_probe_examples(
                train_ds.examples,
                max_scan=32,
                percentile=0.90,
                batch_size=micro_b,
                seed=seed,
                length_fn=_rough_len,
            )
            feats = [train_ds[i] for i in idxs]
            while len(feats) < micro_b:
                feats.append(feats[-1] if feats else train_ds[0])
            batch = collator(feats[:micro_b])
            batch = {k: v.to(device) for k, v in batch.items()}
            model.train()
            with runtime.autocast():
                out = model(**batch)
                loss = out.loss if hasattr(out, "loss") else out["loss"]
            loss.backward()
            model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        try:
            micro_b, _ = auto_tune_micro_batch(
                preferred=preferred,
                world_size=runtime.world_size,
                global_batch_size=sft_cfg.get("global_batch_size")
                or sft_cfg.get("reference_global_batch_size")
                or sft_cfg.get("target_global_batch_size"),
                mode=runtime.mode,
                memory_auto=True,
                probe_fn=probe,
                batch_policy=hw_cfg.get("batch_policy"),
                log=logger.info,
            )
            sft_cfg["per_device_batch_size"] = micro_b
            logger.info(f"[sft][memory-auto] preferred={preferred} selected={micro_b}")
        except Exception as exc:  # noqa: BLE001
            logger.info(f"[sft][memory-auto] probe skipped ({exc})")
            sft_cfg["per_device_batch_size"] = preferred
            dist_utils.barrier("sft-memory-probe-fallback")
    else:
        sft_cfg.setdefault("per_device_batch_size", preferred)

    batch_plan = runtime.resolve_stage_batch("sft", sft_cfg)
    for line in batch_plan.summary_lines():
        logger.info(f"[sft] {line}")

    max_steps = sft_cfg.get("max_steps")

    # DeepSpeed：HF Trainer 原生支持。Do NOT custom-wrap SFT.
    ds_config = _resolve_deepspeed(cfg, logger)
    effective = runtime.effective_strategy or runtime.strategy.strategy

    from llm4rec.runtime.activation_ckpt import resolve_activation_checkpointing

    pressure = getattr(runtime.strategy, "pressure_ratio", None)
    if pressure is None and isinstance(hw_cfg.get("_memory_estimate"), dict):
        pressure = hw_cfg["_memory_estimate"].get("pressure_ratio")
    grad_ckpt, act_reason = resolve_activation_checkpointing(
        hw_cfg,
        preferred_micro=preferred,
        selected_micro=batch_plan.per_device_batch_size,
        pressure_ratio=pressure,
        effective_strategy=effective,
        strategy_source=getattr(runtime.strategy, "source", None),
    )
    cfg.setdefault("hardware", {})["_activation_checkpointing_effective"] = grad_ckpt
    cfg.setdefault("hardware", {})["_activation_checkpointing"] = {
        "requested": hw_cfg.get("activation_checkpointing", "auto"),
        "effective": grad_ckpt,
        "reason": act_reason,
    }

    fsdp_arg: str | list[str] | bool = False
    if ds_config is None and effective == "fsdp" and runtime.world_size > 1:
        fsdp_arg = "full_shard auto_wrap"
        logger.info("[sft] mapping effective strategy=fsdp → TrainingArguments.fsdp")
    elif str(effective).startswith("deepspeed") and ds_config is None:
        logger.warning(
            f"[sft] strategy={effective} but hardware.deepspeed unset; "
            "HF Trainer will use default DDP"
        )

    warmup_kwargs: dict[str, Any] = {}
    if sft_cfg.get("warmup_steps") not in (None, "null", False):
        warmup_kwargs["warmup_steps"] = int(sft_cfg["warmup_steps"])
    else:
        warmup_kwargs["warmup_ratio"] = float(sft_cfg.get("warmup_ratio") or 0.0)

    from llm4rec.runtime.checkpointing import resolve_save_steps, resolve_save_total_limit

    def _step_interval(value: Any) -> int | float | None:
        if value in (None, 0, "null", False):
            return None
        if isinstance(value, float) and 0.0 < value < 1.0:
            return value
        return int(value)

    save_steps = resolve_save_steps(cfg, sft_cfg)
    eval_steps = _step_interval(sft_cfg.get("eval_steps"))
    eval_strategy = "steps" if eval_ds is not None else "no"
    save_strategy = "steps" if save_steps is not None else "no"
    save_total_limit = resolve_save_total_limit(cfg)
    if save_steps is not None:
        logger.info(
            f"[sft] 中间 checkpoint：每 {save_steps} step 存一次"
            f"（最多保留 {save_total_limit} 个）"
        )

    args = TrainingArguments(
        deepspeed=ds_config,
        fsdp=fsdp_arg,
        output_dir=str(output_dir),
        num_train_epochs=float(sft_cfg.get("epochs") or 1),
        max_steps=int(max_steps) if max_steps not in (None, 0, "null") else -1,
        per_device_train_batch_size=batch_plan.per_device_batch_size,
        per_device_eval_batch_size=batch_plan.per_device_batch_size,
        gradient_accumulation_steps=batch_plan.gradient_accumulation_steps,
        learning_rate=float(sft_cfg.get("learning_rate") or 1e-5),
        lr_scheduler_type=str(sft_cfg.get("lr_scheduler_type") or "cosine"),
        weight_decay=float(sft_cfg.get("weight_decay") or 0.0),
        adam_beta1=float(sft_cfg.get("adam_beta1") or 0.9),
        adam_beta2=float(sft_cfg.get("adam_beta2") or 0.999),
        adam_epsilon=float(sft_cfg.get("adam_epsilon") or 1e-8),
        max_grad_norm=float(sft_cfg.get("max_grad_norm") or 1.0),
        logging_steps=int(sft_cfg.get("logging_steps") or 10),
        eval_strategy=eval_strategy,
        eval_steps=eval_steps if eval_ds is not None else None,
        save_strategy=save_strategy,
        save_steps=save_steps if save_steps is not None else 500,
        save_total_limit=save_total_limit,
        load_best_model_at_end=bool(sft_cfg.get("load_best_model_at_end", False))
        and eval_ds is not None
        and save_strategy == "steps",
        metric_for_best_model="loss" if sft_cfg.get("load_best_model_at_end") else None,
        bf16=precision in ("bf16", "bfloat16"),
        fp16=precision in ("fp16", "float16"),
        gradient_checkpointing=grad_ckpt,
        seed=int(cfg.get("seed") or 42),
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=_disable_tqdm(sft_cfg),
        **_length_group_kwargs(sft_cfg),
        **warmup_kwargs,
        **_torch_compile_kwargs(cfg),
    )

    callbacks: list[TrainerCallback] = [MetricsCallback(logger, stage)]
    throughput_cb = ThroughputCallback(cfg)
    callbacks.append(throughput_cb)
    patience = sft_cfg.get("early_stopping_patience")
    if patience and eval_ds is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(patience)))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"[sft] 全参微调：可训练 {trainable / 1e6:.1f}M / 总计 {total / 1e6:.1f}M "
        f"({100 * trainable / max(total, 1):.1f}%)  train={len(train_ds)} eval={len(eval_ds or [])}"
    )
    if reference_sft and hasattr(model, "config"):
        model.config.use_cache = False

    result = trainer.train()

    final_dir = Path(output_dir) / "final"
    # HF Trainer only writes on rank0; other ranks must not race into the next
    # stage (eval) and load a half-written checkpoint.
    if dist_utils.is_main():
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        (Path(output_dir) / "train_log.json").write_text(
            json.dumps(trainer.state.log_history, indent=2) + "\n", encoding="utf-8"
        )
    dist_utils.barrier("sft_saved")

    perf = dict(throughput_cb.metrics) if throughput_cb.metrics else {}
    # Prefer HF train_samples_per_second when available
    for k in ("train_samples_per_second", "train_steps_per_second"):
        if k in result.metrics:
            short = "samples_per_sec" if "samples" in k else "optimizer_steps_per_sec"
            perf[short] = float(result.metrics[k])
    if perf:
        cfg.setdefault("_performance", {})["sft"] = perf

    summary = {
        "stage": stage,
        "checkpoint": str(final_dir),
        "metrics": dict(result.metrics),
        "n_train": len(train_ds),
        "n_eval": len(eval_ds or []),
        "trainable_params": trainable,
        "total_params": total,
        "batch_plan": batch_plan.to_dict(),
        "strategy": runtime.effective_strategy,
        "precision": precision,
        "performance": perf or None,
    }
    logger.info(f"[sft] 完成，权重 → {final_dir}")
    return summary
