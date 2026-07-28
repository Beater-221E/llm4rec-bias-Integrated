"""TTY progress callback for HF Trainer (SFT / GRPO) — one in-place bar."""

from __future__ import annotations

from typing import Any, TextIO

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_callback import PrinterCallback, ProgressCallback


def _progress_stream() -> TextIO | None:
    from llm4rec_bias_Integrated.tracking.inplace_progress import get_progress_stream

    return get_progress_stream()


def attach_training_progress(trainer: Any, *, stage: str) -> None:
    """Remove HF printer/default bars; attach a single TTY progress bar."""
    for cb_cls in (PrinterCallback, ProgressCallback):
        try:
            trainer.pop_callback(cb_cls)
        except Exception:
            try:
                trainer.remove_callback(cb_cls)
            except Exception:
                pass
    trainer.add_callback(TtyTrainingProgressCallback(stage=stage))


class TtyTrainingProgressCallback(TrainerCallback):
    """Progress bar on the line below the shell's stage header."""

    def __init__(self, stage: str = "train") -> None:
        self.stage = stage
        self._total = 0
        self._last = -1
        # Bar label only (sft / grpo); stage name is on the line above.
        self._desc = stage

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not state.is_local_process_zero:
            return
        self._total = int(state.max_steps or 0)
        self._last = -1
        self._draw(0)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not state.is_local_process_zero:
            return
        step = int(state.global_step)
        if step == self._last:
            return
        self._last = step
        if self._total <= 0 and state.max_steps:
            self._total = int(state.max_steps)
        self._draw(step)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        # Some trainers update global_step on log more reliably than step_end.
        if not state.is_local_process_zero:
            return
        step = int(state.global_step or 0)
        if step <= 0 or step == self._last:
            return
        self._last = step
        if self._total <= 0 and state.max_steps:
            self._total = int(state.max_steps)
        self._draw(step)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not state.is_local_process_zero:
            return
        step = int(state.global_step)
        if self._total > 0:
            self._draw(self._total)
        elif step > 0:
            self._draw(step)
        stream = _progress_stream()
        if stream is not None:
            try:
                stream.write("\r\033[K")
                stream.flush()
            except Exception:
                pass

    def _draw(self, step: int) -> None:
        tty = _progress_stream()
        if tty is None:
            return
        total = max(self._total, step, 1)
        step = min(step, total)
        width = 24
        filled = int(width * step / total) if total else 0
        bar = "█" * filled + "░" * (width - filled)
        pct = 100.0 * step / total if total else 0.0
        line = f"{self._desc} |{bar}| {step}/{total} {pct:5.1f}%"
        try:
            tty.write(f"\r\033[K{line}")
            tty.flush()
        except Exception:
            try:
                tty.write(f"\r{line}")
                tty.flush()
            except Exception:
                pass
