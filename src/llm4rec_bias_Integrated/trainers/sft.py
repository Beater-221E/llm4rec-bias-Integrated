"""LoRA SFT trainer for candidate-choice (and compatible chat) tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

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
from llm4rec_bias_Integrated.trainers.base import Trainer
from llm4rec_bias_Integrated.trainers.callbacks import JsonlMetricsCallback, disable_hf_printer
from llm4rec_bias_Integrated.trainers.progress import attach_training_progress
from llm4rec_bias_Integrated.trainers.checkpointing import final_adapter_dir, stage_dir
from llm4rec_bias_Integrated.trainers.distributed import (
    maybe_init_process_group,
    resolve_distributed_plan,
)


def examples_to_sft_dataset(examples: list[RecommendationExample]) -> Dataset:
    """Convert standardized examples to TRL chat ``messages`` rows."""
    rows = []
    for ex in examples:
        messages = list(ex.prompt_messages) + [
            {"role": "assistant", "content": ex.target_text}
        ]
        rows.append({"messages": messages})
    if not rows:
        raise ConfigurationError("SFT dataset is empty — check train_limit / prepare")
    return Dataset.from_list(rows)


class SFTLoRATrainer(Trainer):
    """Supervised fine-tuning with LoRA via ``trl.SFTTrainer``."""

    name = "sft"

    def __init__(
        self,
        train_examples: list[RecommendationExample],
        eval_examples: list[RecommendationExample] | None = None,
    ) -> None:
        self.train_examples = train_examples
        self.eval_examples = eval_examples or []
        self._last_output: Path | None = None
        self._trainer: SFTTrainer | None = None

    def train(self, context: ExperimentContext) -> dict[str, Any]:
        require_cuda()
        cfg = context.config
        training = cfg.get("training") or {}
        model_cfg = cfg.get("model") or {}
        peft_cfg = cfg.get("peft") or {}

        plan = resolve_distributed_plan(training, model_name=str(model_cfg.get("name")))
        maybe_init_process_group(plan)
        precision = resolve_precision(str(model_cfg.get("dtype") or "auto"))
        preflight = hardware_preflight(str(model_cfg.get("checkpoint")), precision)
        preflight["distributed"] = {
            "strategy": plan.strategy,
            "world_size": plan.world_size,
            "nproc_per_node": plan.nproc_per_node,
            "effective_batch_size": plan.effective_batch_size,
            "launch_hint": plan.launch_hint,
        }

        train_ds = examples_to_sft_dataset(self.train_examples)
        eval_ds = (
            examples_to_sft_dataset(self.eval_examples)
            if self.eval_examples
            else None
        )

        out_root = stage_dir(context.run_dir, "sft")
        final_dir = final_adapter_dir(context.run_dir, "sft")

        if not peft_cfg.get("enabled", True):
            raise ConfigurationError(
                "Phase-3 candidate-choice SFT expects peft.enabled=true (LoRA)."
            )
        lora: LoraConfig = build_lora_config(peft_cfg)

        max_steps = training.get("max_steps")
        sft_args = SFTConfig(
            output_dir=str(out_root),
            num_train_epochs=float(training.get("epochs", 1)),
            max_steps=int(max_steps) if max_steps not in (None, 0, "null") else -1,
            per_device_train_batch_size=int(training.get("batch_size", 2)),
            per_device_eval_batch_size=int(
                training.get("eval_batch_size", training.get("batch_size", 2))
            ),
            gradient_accumulation_steps=int(
                training.get("gradient_accumulation_steps", 8)
            ),
            learning_rate=float(training.get("learning_rate", 1e-4)),
            lr_scheduler_type=str(training.get("lr_scheduler_type", "cosine")),
            warmup_ratio=float(training.get("warmup_ratio", 0.03)),
            logging_steps=int(training.get("logging_steps", 50)),
            eval_strategy="steps" if eval_ds is not None else "no",
            eval_steps=int(training.get("eval_steps", 50)) if eval_ds is not None else None,
            save_strategy="steps",
            save_steps=int(training.get("save_steps", 50)),
            save_total_limit=int(training.get("save_total_limit", 2)),
            max_length=int(training.get("max_seq_length", 1024)),
            # Qwen Instruct templates lack `{% generation %}`, so TRL's
            # assistant_only_loss masks every token → ~0 loss / empty grads.
            # Opt in only when the tokenizer template supports generation spans.
            assistant_only_loss=bool(training.get("assistant_only_loss", False)),
            # TRL 1.9 defaults to chunked_nll; with transformers 5.x, model.forward may
            # already be functools.partial and _patch_chunked_ce_lm_head crashes on __func__.
            loss_type="nll",
            bf16=precision.bf16,
            fp16=precision.fp16,
            gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)),
            report_to=["tensorboard"] if (cfg.get("tracking") or {}).get("tensorboard") else "none",
            seed=int(context.seed),
            data_seed=int(context.seed),
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            dataloader_num_workers=int(training.get("dataloader_num_workers", 0)),
            model_init_kwargs={"torch_dtype": precision.dtype},
            disable_tqdm=bool(training.get("disable_tqdm", False)),
        )

        if plan.is_main_process:
            context.logger.print_startup_summary(
                context.summary_lines()
                + [
                    f"Train examples    : {len(self.train_examples)}",
                    f"Eval examples     : {len(self.eval_examples)}",
                    f"Distributed       : {plan.strategy} (world={plan.world_size})",
                    f"Effective batch   : {plan.effective_batch_size}",
                    f"GPU count         : {preflight['gpu_count']}",
                    f"GPU names         : {preflight['gpu_names']}",
                    f"Precision         : {preflight['dtype']} (bf16={precision.bf16})",
                    f"Launch hint       : {plan.launch_hint}",
                ]
            )
            write_json(context.run_dir / "preflight.json", preflight)

        trainer = SFTTrainer(
            model=str(model_cfg["checkpoint"]),
            args=sft_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=lora,
            callbacks=[JsonlMetricsCallback(context.logger, stage="sft")],
        )
        disable_hf_printer(trainer)
        attach_training_progress(trainer, stage="sft")
        self._trainer = trainer

        # Parameter counts after LoRA wrap
        param_info = count_parameters(trainer.model)
        if plan.is_main_process:
            context.logger.info(
                f"Parameters total={param_info['total_parameters']:,} "
                f"trainable={param_info['trainable_parameters']:,} "
                f"({param_info['trainable_pct']:.2f}%)"
            )
            write_json(context.run_dir / "parameter_counts.json", param_info)

        from llm4rec_bias_Integrated.tracking.inplace_progress import write_progress_status

        write_progress_status("sft: training…")
        train_result = trainer.train()
        if plan.is_main_process:
            trainer.save_model(str(final_dir))
            if hasattr(trainer, "tokenizer") and trainer.tokenizer is not None:
                trainer.tokenizer.save_pretrained(str(final_dir))
            elif (
                hasattr(trainer, "processing_class")
                and trainer.processing_class is not None
            ):
                trainer.processing_class.save_pretrained(str(final_dir))
            # Persist HF log history
            history_path = out_root / "train_log.json"
            history_path.write_text(
                json.dumps(trainer.state.log_history, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            history_path = out_root / "train_log.json"
        self._last_output = final_dir

        metrics = dict(train_result.metrics)
        summary = {
            "stage": "sft",
            "adapter_path": str(final_dir),
            "train_log": str(history_path),
            "metrics": metrics,
            "parameters": param_info,
            "distributed": preflight["distributed"],
            "n_train": len(self.train_examples),
            "n_eval": len(self.eval_examples),
        }
        if plan.is_main_process:
            write_json(out_root / "summary.json", summary)
            context.logger.print_stage_summary(
                [
                    ("train/loss", metrics.get("train_loss", "—"), metrics.get("train_loss", "—"), "final"),
                    ("train/runtime", metrics.get("train_runtime", "—"), metrics.get("train_runtime", "—"), "final"),
                    ("train/samples_per_s", metrics.get("train_samples_per_second", "—"), metrics.get("train_samples_per_second", "—"), "final"),
                ],
                title="SFT stage summary",
            )
            context.logger.log_metrics(
                {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                stage="sft",
                step=int(trainer.state.global_step),
                split="train",
            )
        return summary

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        if self._trainer is None:
            raise ConfigurationError("No trainer to save — call train() first")
        self._trainer.save_model(str(output_dir))
        self._last_output = output_dir
        return output_dir
