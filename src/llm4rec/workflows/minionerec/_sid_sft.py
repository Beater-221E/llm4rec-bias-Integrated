"""SID-route SFT trainer (generative retrieval)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from peft import get_peft_model
from trl import SFTConfig, SFTTrainer

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
from llm4rec.workflows.minionerec.semantic_ids.build import load_sid_jsonl, sid_dir
from llm4rec.workflows.minionerec.semantic_ids.table import SidTable
from llm4rec.components.trainer._impl.base import Trainer
from llm4rec.components.trainer._impl.callbacks import JsonlMetricsCallback, disable_hf_printer
from llm4rec.components.trainer._impl.progress import attach_training_progress
from llm4rec.components.trainer._impl.checkpointing import final_adapter_dir, stage_dir
from llm4rec.components.trainer._impl.distributed import (
    maybe_init_process_group,
    resolve_distributed_plan,
)


class SidSFTTrainer(Trainer):
    name = "sid_sft"

    def __init__(self, *, processed_dir: Path) -> None:
        self.processed_dir = Path(processed_dir)
        self._last_output: Path | None = None

    def train(self, context: ExperimentContext) -> dict[str, Any]:
        require_cuda()
        cfg = context.config
        training = cfg.get("training") or {}
        model_cfg = cfg.get("model") or {}
        peft_cfg = dict(cfg.get("peft") or {})
        ds_cfg = cfg.get("dataset") or {}

        plan = resolve_distributed_plan(training, model_name=str(model_cfg.get("name")))
        maybe_init_process_group(plan)
        precision = resolve_precision(str(model_cfg.get("dtype") or "auto"))
        preflight = hardware_preflight(str(model_cfg.get("checkpoint")), precision)

        out_sid = sid_dir(self.processed_dir)
        table_path = out_sid / "semantic_ids.json"
        train_path = out_sid / "sid_train.jsonl"
        val_path = out_sid / "sid_val.jsonl"
        if not table_path.is_file() or not train_path.is_file():
            raise MissingArtifactError(
                f"SID artifacts missing under {out_sid}; run prepare with workflow=minionerec"
            )
        table = SidTable(table_path)
        tok, model, new_ids = prepare_sid_model(
            str(model_cfg["checkpoint"]),
            table,
            dtype=str(model_cfg.get("dtype") or "auto"),
            local_rank=plan.local_rank,
        )

        train_limit = ds_cfg.get("train_limit")
        eval_limit = ds_cfg.get("eval_limit")
        train_rows = load_sid_jsonl(train_path, limit=train_limit)
        val_rows = load_sid_jsonl(val_path, limit=eval_limit) if val_path.is_file() else []
        if not train_rows:
            raise ConfigurationError("SID train jsonl is empty")

        def to_messages(rows: list[dict[str, Any]]) -> Dataset:
            return Dataset.from_list(
                [
                    {
                        "messages": list(r["prompt"])
                        + [{"role": "assistant", "content": r["answer"]}]
                    }
                    for r in rows
                ]
            )

        train_ds = to_messages(train_rows)
        eval_ds = to_messages(val_rows) if val_rows else None

        # Broader targets for SID (match upstream)
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
        lora = build_lora_config(
            peft_cfg, trainable_token_indices={"embed_tokens": new_ids}
        )
        model = get_peft_model(model, lora)
        head = model.get_output_embeddings()
        if not hasattr(head, "bias"):
            head.bias = None

        out_root = stage_dir(context.run_dir, "sft")
        final_dir = final_adapter_dir(context.run_dir, "sft")
        max_steps = training.get("max_steps")

        args = SFTConfig(
            output_dir=str(out_root),
            num_train_epochs=float(training.get("epochs", 1)),
            max_steps=int(max_steps) if max_steps not in (None, 0, "null") else -1,
            per_device_train_batch_size=int(training.get("batch_size", 2)),
            gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 2)),
            learning_rate=float(training.get("learning_rate", 1e-4)),
            logging_steps=int(training.get("logging_steps", 50)),
            eval_strategy="steps" if eval_ds is not None else "no",
            eval_steps=int(training.get("eval_steps", 50)) if eval_ds is not None else None,
            save_steps=int(training.get("save_steps", 50)),
            save_total_limit=int(training.get("save_total_limit", 2)),
            max_length=int(training.get("max_seq_length", 1024)),
            # Same as letter SFT: Qwen templates lack `{% generation %}`.
            assistant_only_loss=bool(training.get("assistant_only_loss", False)),
            # Avoid TRL chunked_nll patch crash when model.forward is functools.partial.
            loss_type="nll",
            bf16=precision.bf16,
            fp16=precision.fp16,
            report_to="none",
            seed=int(context.seed),
            disable_tqdm=bool(training.get("disable_tqdm", False)),
        )
        if plan.is_main_process:
            context.logger.print_startup_summary(
                context.summary_lines()
                + [
                    "Stage             : sid_sft",
                    f"Train rows        : {len(train_ds)}",
                    f"SID levels        : {table.levels}",
                    f"New tokens        : {len(new_ids)}",
                    f"GPU               : {preflight['gpu_names']}",
                ]
            )

        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tok,
            callbacks=[JsonlMetricsCallback(context.logger, stage="sft")],
        )
        disable_hf_printer(trainer)
        attach_training_progress(trainer, stage="sft")
        param_info = count_parameters(trainer.model)
        from llm4rec.tracking.inplace_progress import write_progress_status

        write_progress_status("sft: training…")
        result = trainer.train()
        if plan.is_main_process:
            trainer.save_model(str(final_dir))
            tok.save_pretrained(str(final_dir))
            history_path = out_root / "train_log.json"
            history_path.write_text(
                json.dumps(trainer.state.log_history, indent=2) + "\n", encoding="utf-8"
            )
            write_json(
                out_root / "summary.json",
                {
                    "stage": "sid_sft",
                    "adapter_path": str(final_dir),
                    "train_log": str(history_path),
                    "metrics": dict(result.metrics),
                    "parameters": param_info,
                    "n_train": len(train_ds),
                    "sid_table": str(table_path),
                },
            )
        self._last_output = final_dir
        return {
            "stage": "sid_sft",
            "adapter_path": str(final_dir),
            "train_log": str(out_root / "train_log.json"),
            "metrics": dict(result.metrics),
            "parameters": param_info,
            "n_train": len(train_ds),
            "sid_table": str(table_path),
        }

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
