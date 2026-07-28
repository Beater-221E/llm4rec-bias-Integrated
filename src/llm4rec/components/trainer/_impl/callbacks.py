"""Training callbacks — metrics → JSONL + file log tables (not terminal)."""

from __future__ import annotations

from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_callback import PrinterCallback

from llm4rec.tracking.logger import ExperimentLogger

# Prefer a short, readable subset for periodic log tables.
_STEP_KEYS = [
    "loss",
    "grad_norm",
    "learning_rate",
    "reward",
    "rewards/SidPrefixReward/mean",
    "kl",
    "entropy",
    "shortcut/invalid_rate",
    "completions/mean_length",
    "mean_token_accuracy",
    "epoch",
    "eval_loss",
    "eval_mean_token_accuracy",
]


def disable_hf_printer(trainer: Any) -> None:
    """Remove transformers' default PrinterCallback (raw dict dumps)."""
    try:
        trainer.pop_callback(PrinterCallback)
    except Exception:
        try:
            trainer.remove_callback(PrinterCallback)
        except Exception:
            pass


class JsonlMetricsCallback(TrainerCallback):
    """Mirror HF trainer logs into metrics.jsonl + run console.log tables.

    Terminal stays clean (tqdm progress bar only). Stats tables go to the
    run log file on each ``logging_steps`` tick (default 50).
    """

    def __init__(self, logger: ExperimentLogger, stage: str = "sft") -> None:
        self.logger = logger
        self.stage = stage

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs or not state.is_local_process_zero:
            return

        metrics = {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in logs.items()
            if k != "total_flos"
        }
        step = int(state.global_step)
        self.logger.log_metrics(
            metrics,
            stage=self.stage,
            step=step,
            epoch=float(state.epoch) if state.epoch is not None else None,
            split="train" if "loss" in logs and "eval_loss" not in logs else "eval",
        )
        preferred = [k for k in _STEP_KEYS if k in metrics]
        extras = [k for k in metrics if k not in preferred]
        show_keys = preferred + extras[:8]
        title = f"{self.stage} step {step}"
        # File only — do not print to terminal (keeps single-line tqdm visible).
        self.logger.write_metrics_table(metrics, title=title, keys=show_keys)
