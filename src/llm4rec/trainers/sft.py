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
    """右侧 padding；label 用 -100 填充。"""

    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch: dict[str, list[list[int]]] = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
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
) -> dict[str, Any]:
    """跑全参 SFT，把完整权重存到 ``output_dir/final``。"""
    sft_cfg = (cfg.get("train") or {}).get("sft") or {}
    if not train_examples:
        raise ConfigurationError("SFT 训练集为空")

    max_len = int(sft_cfg.get("max_seq_length") or 1024)
    train_ds = ChatSFTDataset(train_examples, tokenizer, max_len)
    eval_ds = ChatSFTDataset(eval_examples, tokenizer, max_len) if eval_examples else None

    precision = str(cfg["hardware"].get("precision") or "fp32").lower()
    max_steps = sft_cfg.get("max_steps")

    # DeepSpeed：HF Trainer 原生支持，传 json 路径即可，"auto" 字段由它按
    # TrainingArguments 自动填。RL/DPO 是手写循环，走 DDP（见 core/distributed.py）。
    ds_config = _resolve_deepspeed(cfg, logger)

    args = TrainingArguments(
        deepspeed=ds_config,
        output_dir=str(output_dir),
        num_train_epochs=float(sft_cfg.get("epochs") or 1),
        max_steps=int(max_steps) if max_steps not in (None, 0, "null") else -1,
        per_device_train_batch_size=int(sft_cfg.get("per_device_batch_size") or 2),
        per_device_eval_batch_size=int(sft_cfg.get("per_device_batch_size") or 2),
        gradient_accumulation_steps=int(sft_cfg.get("gradient_accumulation_steps") or 1),
        learning_rate=float(sft_cfg.get("learning_rate") or 1e-5),
        lr_scheduler_type=str(sft_cfg.get("lr_scheduler_type") or "cosine"),
        warmup_ratio=float(sft_cfg.get("warmup_ratio") or 0.0),
        weight_decay=float(sft_cfg.get("weight_decay") or 0.0),
        adam_beta1=float(sft_cfg.get("adam_beta1") or 0.9),
        adam_beta2=float(sft_cfg.get("adam_beta2") or 0.999),
        adam_epsilon=float(sft_cfg.get("adam_epsilon") or 1e-8),
        max_grad_norm=float(sft_cfg.get("max_grad_norm") or 1.0),
        logging_steps=int(sft_cfg.get("logging_steps") or 10),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=int(sft_cfg.get("eval_steps") or 200) if eval_ds is not None else None,
        # checkpoint 策略：默认只在 stage 结束存一份完整权重。
        # 0.5B 全参一份 ~2GB，按 step 存会迅速吃满盘。
        save_strategy="steps" if cfg["checkpoint"].get("save_steps") else "no",
        save_steps=int(cfg["checkpoint"].get("save_steps") or 0) or None,
        save_total_limit=int(cfg["checkpoint"].get("save_total_limit") or 1),
        load_best_model_at_end=bool(sft_cfg.get("load_best_model_at_end", False))
        and eval_ds is not None
        and bool(cfg["checkpoint"].get("save_steps")),
        bf16=precision in ("bf16", "bfloat16"),
        fp16=precision in ("fp16", "float16"),
        gradient_checkpointing=bool(cfg["hardware"].get("gradient_checkpointing", False)),
        seed=int(cfg.get("seed") or 42),
        # wandb 由我们自己的 logger 统一推，不让 HF 再开一个 run
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=False,
    )

    callbacks: list[TrainerCallback] = [MetricsCallback(logger, stage)]
    patience = sft_cfg.get("early_stopping_patience")
    if patience and eval_ds is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(patience)))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PadCollator(tokenizer.pad_token_id or tokenizer.eos_token_id or 0),
        callbacks=callbacks,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"[sft] 全参微调：可训练 {trainable / 1e6:.1f}M / 总计 {total / 1e6:.1f}M "
        f"({100 * trainable / max(total, 1):.1f}%)  train={len(train_ds)} eval={len(eval_ds or [])}"
    )

    result = trainer.train()

    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    (Path(output_dir) / "train_log.json").write_text(
        json.dumps(trainer.state.log_history, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "stage": stage,
        "checkpoint": str(final_dir),
        "metrics": dict(result.metrics),
        "n_train": len(train_ds),
        "n_eval": len(eval_ds or []),
        "trainable_params": trainable,
        "total_params": total,
    }
    logger.info(f"[sft] 完成，权重 → {final_dir}")
    return summary
