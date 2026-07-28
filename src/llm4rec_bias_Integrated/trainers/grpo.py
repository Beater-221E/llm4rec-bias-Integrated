"""GRPO trainer for recommendation (candidate-choice Phase 5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from llm4rec_bias_Integrated.core.context import ExperimentContext
from llm4rec_bias_Integrated.core.exceptions import ConfigurationError
from llm4rec_bias_Integrated.core.reproducibility import write_json
from llm4rec_bias_Integrated.core.schemas import RecommendationExample
from llm4rec_bias_Integrated.models.base import (
    count_parameters,
    hardware_preflight,
    require_cuda,
    resolve_precision,
)
from llm4rec_bias_Integrated.models.peft import build_lora_config
from llm4rec_bias_Integrated.rewards.composer import build_trl_reward_from_config
from llm4rec_bias_Integrated.trainers.base import Trainer
from llm4rec_bias_Integrated.trainers.callbacks import JsonlMetricsCallback, disable_hf_printer
from llm4rec_bias_Integrated.trainers.progress import attach_training_progress
from llm4rec_bias_Integrated.trainers.checkpointing import final_adapter_dir, stage_dir
from llm4rec_bias_Integrated.trainers.distributed import (
    maybe_init_process_group,
    resolve_distributed_plan,
    wait_for_file,
)


def examples_to_grpo_dataset(examples: list[RecommendationExample]) -> Dataset:
    """Build a TRL GRPO dataset: prompt messages + target + pop_quantiles."""
    rows = []
    for ex in examples:
        if ex.target_index is None or not ex.candidates:
            continue
        rows.append(
            {
                "prompt": list(ex.prompt_messages),
                "target": int(ex.target_index),
                "pop_quantiles": list(ex.features.get("pop_quantiles") or []),
            }
        )
    if not rows:
        raise ConfigurationError("GRPO dataset is empty")
    return Dataset.from_list(rows)


class GRPOLoRATrainer(Trainer):
    """Group-relative policy optimization with composable rewards."""

    name = "grpo"

    def __init__(
        self,
        train_examples: list[RecommendationExample],
        *,
        sft_adapter_path: str | None = None,
    ) -> None:
        self.train_examples = train_examples
        self.sft_adapter_path = sft_adapter_path
        self._last_output: Path | None = None
        self._trainer: GRPOTrainer | None = None

    def train(self, context: ExperimentContext) -> dict[str, Any]:
        require_cuda()
        cfg = context.config
        training = cfg.get("training") or {}
        model_cfg = cfg.get("model") or {}
        peft_cfg = cfg.get("peft") or {}
        grpo_cfg = cfg.get("grpo") or {}

        num_generations = int(grpo_cfg.get("num_generations", 4))
        if num_generations < 2:
            raise ConfigurationError("grpo.num_generations must be >= 2")

        plan = resolve_distributed_plan(training, model_name=str(model_cfg.get("name")))
        maybe_init_process_group(plan)
        precision = resolve_precision(str(model_cfg.get("dtype") or "auto"))
        preflight = hardware_preflight(str(model_cfg.get("checkpoint")), precision)

        # Resolve init checkpoint: explicit > SFT final in this run > config
        init_ckpt = (
            cfg.get("init_checkpoint")
            or self.sft_adapter_path
            or training.get("init_checkpoint")
        )

        # Load base + optional merge SFT (spec: merge then new LoRA for GRPO)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_cfg["checkpoint"]),
                dtype=precision.dtype,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_cfg["checkpoint"]),
                torch_dtype=precision.dtype,
            )
        model = model.to(f"cuda:{plan.local_rank}")
        if init_ckpt:
            wait_for_file(Path(init_ckpt) / "adapter_config.json")
            model = PeftModel.from_pretrained(model, str(init_ckpt)).merge_and_unload()
            if plan.is_main_process:
                context.logger.info(f"Merged init adapter for GRPO: {init_ckpt}")

        if not peft_cfg.get("enabled", True):
            raise ConfigurationError("GRPO Phase 5 expects peft.enabled=true")
        lora: LoraConfig = build_lora_config(peft_cfg)

        train_ds = examples_to_grpo_dataset(self.train_examples)
        reward_fn = build_trl_reward_from_config(grpo_cfg)

        out_root = stage_dir(context.run_dir, "grpo")
        final_dir = final_adapter_dir(context.run_dir, "grpo")

        prompts_per_step = int(
            grpo_cfg.get("prompts_per_step")
            or training.get("prompts_per_step")
            or max(1, int(training.get("batch_size", 1)))
        )
        # TRL expects per_device_train_batch_size divisible by num_generations
        per_device = prompts_per_step * num_generations

        max_steps = int(
            grpo_cfg.get("max_steps")
            or training.get("max_steps")
            or training.get("grpo_steps")
            or 100
        )

        args = GRPOConfig(
            output_dir=str(out_root),
            max_steps=max_steps,
            per_device_train_batch_size=per_device,
            num_generations=num_generations,
            max_completion_length=int(grpo_cfg.get("max_completion_length", 16)),
            temperature=float(grpo_cfg.get("temperature", 1.0)),
            top_p=float(grpo_cfg.get("top_p", 1.0)),
            beta=float(grpo_cfg.get("beta", 0.04)),
            scale_rewards=str(grpo_cfg.get("advantage_normalization", "group")),
            learning_rate=float(
                grpo_cfg.get("learning_rate")
                or training.get("learning_rate")
                or 5e-6
            ),
            logging_steps=int(training.get("logging_steps", 50)),
            save_steps=int(training.get("save_steps", 50)),
            save_total_limit=int(training.get("save_total_limit", 2)),
            bf16=precision.bf16,
            fp16=precision.fp16,
            report_to=["tensorboard"]
            if (cfg.get("tracking") or {}).get("tensorboard")
            else "none",
            seed=int(context.seed),
            remove_unused_columns=False,
            gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)),
            disable_tqdm=bool(training.get("disable_tqdm", False)),
        )

        if plan.is_main_process:
            context.logger.print_startup_summary(
                context.summary_lines()
                + [
                    f"Stage             : grpo",
                    f"Init adapter      : {init_ckpt}",
                    f"Train prompts     : {len(train_ds)}",
                    f"num_generations   : {num_generations}",
                    f"beta (KL)         : {args.beta}",
                    f"max_steps         : {max_steps}",
                    f"per_device_batch  : {per_device}",
                    f"Reward weights    : {grpo_cfg.get('reward_weights')}",
                    f"Distributed       : {plan.strategy} (world={plan.world_size})",
                    f"GPU count         : {preflight['gpu_count']}",
                ]
            )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_fn,
            args=args,
            train_dataset=train_ds,
            peft_config=lora,
            callbacks=[JsonlMetricsCallback(context.logger, stage="grpo")],
        )
        disable_hf_printer(trainer)
        attach_training_progress(trainer, stage="grpo")
        self._trainer = trainer
        param_info = count_parameters(trainer.model)
        if plan.is_main_process:
            write_json(context.run_dir / "parameter_counts_grpo.json", param_info)

        from llm4rec_bias_Integrated.tracking.inplace_progress import write_progress_status

        write_progress_status("grpo: training…")
        result = trainer.train()
        history_path = out_root / "train_log.json"
        if plan.is_main_process:
            trainer.save_model(str(final_dir))
            if (
                hasattr(trainer, "processing_class")
                and trainer.processing_class is not None
            ):
                trainer.processing_class.save_pretrained(str(final_dir))
            history_path.write_text(
                json.dumps(trainer.state.log_history, indent=2) + "\n",
                encoding="utf-8",
            )
        self._last_output = final_dir

        metrics = dict(result.metrics)
        # Extract last logged reward / kl from history when present
        last_reward = None
        last_kl = None
        for row in reversed(trainer.state.log_history):
            if last_reward is None and "reward" in row:
                last_reward = float(row["reward"])
            if last_kl is None and "kl" in row:
                last_kl = float(row["kl"])
            if last_reward is not None and last_kl is not None:
                break

        summary = {
            "stage": "grpo",
            "adapter_path": str(final_dir),
            "init_checkpoint": str(init_ckpt) if init_ckpt else None,
            "train_log": str(history_path),
            "metrics": metrics,
            "last_reward": last_reward,
            "last_kl": last_kl,
            "parameters": param_info,
            "n_train": len(train_ds),
            "num_generations": num_generations,
            "beta": float(args.beta),
        }
        if plan.is_main_process:
            write_json(out_root / "summary.json", summary)
            context.logger.print_stage_summary(
                [
                    ("train/reward", last_reward, last_reward, "final"),
                    ("train/kl", last_kl, last_kl, "final"),
                    ("train/loss", metrics.get("train_loss", "—"), metrics.get("train_loss", "—"), "final"),
                ],
                title="GRPO stage summary",
            )
            context.logger.log_metrics(
                {
                    k: float(v)
                    for k, v in {
                        **{kk: vv for kk, vv in metrics.items() if isinstance(vv, (int, float))},
                        "reward": last_reward,
                        "kl": last_kl,
                    }.items()
                    if isinstance(v, (int, float))
                },
                stage="grpo",
                step=int(trainer.state.global_step),
                split="train",
            )
        return summary

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        if self._trainer is None:
            raise ConfigurationError("No GRPO trainer to save")
        self._trainer.save_model(str(output_dir))
        return output_dir
