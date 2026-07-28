"""SID-route GRPO trainer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from peft import LoraConfig, PeftModel
from trl import GRPOConfig, GRPOTrainer

from llm4rec.core.context import ExperimentContext
from llm4rec.core.exceptions import ConfigurationError, MissingArtifactError
from llm4rec.core.reproducibility import write_json
from llm4rec.components.model._impl.base import (
    count_parameters,
    hardware_preflight,
    require_cuda,
    resolve_precision,
)
from llm4rec.components.model._impl.peft import build_lora_config
from llm4rec.components.model._impl.sid import prepare_sid_model
from llm4rec.components.reward._impl.sid_prefix import build_trl_sid_prefix_reward
from llm4rec.workflows.minionerec.semantic_ids.build import load_sid_jsonl, sid_dir
from llm4rec.workflows.minionerec.semantic_ids.table import SidTable
from llm4rec.components.trainer._impl.base import Trainer
from llm4rec.components.trainer._impl.callbacks import JsonlMetricsCallback, disable_hf_printer
from llm4rec.components.trainer._impl.progress import attach_training_progress
from llm4rec.components.trainer._impl.checkpointing import final_adapter_dir, stage_dir
from llm4rec.components.trainer._impl.distributed import (
    maybe_init_process_group,
    resolve_distributed_plan,
    wait_for_file,
)


class SidGRPOTrainer(Trainer):
    name = "sid_grpo"

    def __init__(
        self,
        *,
        processed_dir: Path,
        sft_adapter_path: str | None = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.sft_adapter_path = sft_adapter_path
        self._last_output: Path | None = None

    def train(self, context: ExperimentContext) -> dict[str, Any]:
        require_cuda()
        cfg = context.config
        training = cfg.get("training") or {}
        model_cfg = cfg.get("model") or {}
        peft_cfg = dict(cfg.get("peft") or {})
        grpo_cfg = cfg.get("grpo") or {}
        ds_cfg = cfg.get("dataset") or {}

        num_generations = int(grpo_cfg.get("num_generations", 4))
        if num_generations < 2:
            raise ConfigurationError("grpo.num_generations must be >= 2")

        plan = resolve_distributed_plan(training, model_name=str(model_cfg.get("name")))
        maybe_init_process_group(plan)
        precision = resolve_precision(str(model_cfg.get("dtype") or "auto"))
        preflight = hardware_preflight(str(model_cfg.get("checkpoint")), precision)

        out_sid = sid_dir(self.processed_dir)
        table_path = out_sid / "semantic_ids.json"
        meta_path = out_sid / "item_meta.json"
        train_path = out_sid / "sid_train.jsonl"
        for p in (table_path, meta_path, train_path):
            if not p.is_file():
                raise MissingArtifactError(f"missing {p}")

        init_ckpt = (
            self.sft_adapter_path
            or cfg.get("init_checkpoint")
            or training.get("init_checkpoint")
        )
        if not init_ckpt:
            raise ConfigurationError("SID GRPO requires SFT adapter (init_checkpoint)")

        table = SidTable(table_path)
        tok, model, _ = prepare_sid_model(
            str(model_cfg["checkpoint"]),
            table,
            dtype=str(model_cfg.get("dtype") or "auto"),
            local_rank=plan.local_rank,
        )
        wait_for_file(Path(init_ckpt) / "adapter_config.json")
        model = PeftModel.from_pretrained(model, str(init_ckpt)).merge_and_unload()
        if plan.is_main_process:
            context.logger.info(f"Merged SID SFT adapter: {init_ckpt}")

        if not peft_cfg.get("target_modules"):
            peft_cfg["target_modules"] = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        peft_cfg["dropout"] = float(peft_cfg.get("dropout", 0.0))
        lora: LoraConfig = build_lora_config(peft_cfg)

        train_limit = ds_cfg.get("train_limit")
        rows = load_sid_jsonl(train_path, limit=train_limit)
        ds = Dataset.from_list(
            [
                {
                    "prompt": r["prompt"],
                    "target_item": str(r["target_item"]),
                    "hist_pop_mean": float(r.get("hist_pop_mean", 0.5)),
                }
                for r in rows
            ]
        )

        reward_fn = build_trl_sid_prefix_reward(
            sid_table_path=str(table_path),
            item_meta_path=str(meta_path),
            prefix_credit=float(grpo_cfg.get("prefix_credit", 0.1)),
            invalid_penalty=float(grpo_cfg.get("invalid_penalty", -0.5)),
        )

        out_root = stage_dir(context.run_dir, "grpo")
        final_dir = final_adapter_dir(context.run_dir, "grpo")
        prompts_per_step = int(
            grpo_cfg.get("prompts_per_step")
            or training.get("prompts_per_step")
            or max(1, int(training.get("batch_size", 1)))
        )
        per_device = prompts_per_step * num_generations
        max_steps = int(
            grpo_cfg.get("max_steps") or training.get("max_steps") or 100
        )

        args = GRPOConfig(
            output_dir=str(out_root),
            max_steps=max_steps,
            per_device_train_batch_size=per_device,
            num_generations=num_generations,
            max_completion_length=int(grpo_cfg.get("max_completion_length", 8)),
            temperature=float(grpo_cfg.get("temperature", 0.9)),
            top_p=float(grpo_cfg.get("top_p", 1.0)),
            beta=float(grpo_cfg.get("beta", 0.04)),
            scale_rewards=str(grpo_cfg.get("advantage_normalization", "group")),
            learning_rate=float(
                grpo_cfg.get("learning_rate") or training.get("learning_rate") or 5e-6
            ),
            logging_steps=int(training.get("logging_steps", 50)),
            save_steps=int(training.get("save_steps", 50)),
            save_total_limit=int(training.get("save_total_limit", 2)),
            bf16=precision.bf16,
            fp16=precision.fp16,
            report_to="none",
            seed=int(context.seed),
            remove_unused_columns=False,
            gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)),
            disable_tqdm=bool(training.get("disable_tqdm", False)),
        )

        if plan.is_main_process:
            context.logger.print_startup_summary(
                context.summary_lines()
                + [
                    "Stage             : sid_grpo",
                    f"Init adapter      : {init_ckpt}",
                    f"Train prompts     : {len(ds)}",
                    f"num_generations   : {num_generations}",
                    f"max_steps         : {max_steps}",
                    f"GPU               : {preflight['gpu_names']}",
                ]
            )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_fn,
            args=args,
            train_dataset=ds,
            peft_config=lora,
            processing_class=tok,
            callbacks=[JsonlMetricsCallback(context.logger, stage="grpo")],
        )
        disable_hf_printer(trainer)
        attach_training_progress(trainer, stage="grpo")
        # TRL applies peft_config internally
        param_info = count_parameters(trainer.model)
        from llm4rec.tracking.inplace_progress import write_progress_status

        write_progress_status("grpo: training…")
        result = trainer.train()
        history_path = out_root / "train_log.json"
        if plan.is_main_process:
            trainer.save_model(str(final_dir))
            tok.save_pretrained(str(final_dir))
            history_path.write_text(
                json.dumps(trainer.state.log_history, indent=2) + "\n", encoding="utf-8"
            )
        self._last_output = final_dir

        last_reward = None
        last_kl = None
        last_invalid = None
        for row in reversed(trainer.state.log_history):
            if last_reward is None and "reward" in row:
                last_reward = float(row["reward"])
            if last_kl is None and "kl" in row:
                last_kl = float(row["kl"])
            if last_invalid is None and "shortcut/invalid_rate" in row:
                last_invalid = float(row["shortcut/invalid_rate"])
            if (
                last_reward is not None
                and last_kl is not None
                and last_invalid is not None
            ):
                break

        metrics = dict(result.metrics)
        summary = {
            "stage": "sid_grpo",
            "adapter_path": str(final_dir),
            "init_checkpoint": str(init_ckpt),
            "train_log": str(history_path),
            "metrics": metrics,
            "last_reward": last_reward,
            "last_kl": last_kl,
            "parameters": param_info,
            "n_train": len(ds),
            "sid_table": str(table_path),
        }
        if plan.is_main_process:
            write_json(out_root / "summary.json", summary)
            context.logger.print_stage_summary(
                [
                    ("train/reward", last_reward, last_reward, "final"),
                    ("train/kl", last_kl, last_kl, "final"),
                    ("invalid_rate", last_invalid, last_invalid, "final"),
                    (
                        "train/loss",
                        metrics.get("train_loss", "—"),
                        metrics.get("train_loss", "—"),
                        "final",
                    ),
                ],
                title="SID GRPO stage summary",
            )
        return summary

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
