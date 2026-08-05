"""Experiment logging backends: console + JSONL + wandb."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from llm4rec.tracking.wandb_backend import NullRun, WandbRun, build_wandb_run


class ExperimentLogger:
    """Fan-out logger for metrics and human-readable progress."""

    def __init__(
        self,
        *,
        run_dir: Path | None,
        experiment_id: str = "",
        console_enabled: bool = True,
        jsonl_enabled: bool = True,
        level: int = logging.INFO,
        wandb_run: WandbRun | NullRun | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.experiment_id = experiment_id
        self.wandb = wandb_run or NullRun()
        self.console = Console(stderr=False)
        self._jsonl_path = (run_dir / "metrics.jsonl") if run_dir and jsonl_enabled else None
        self._console_log_path = (run_dir / "console.log") if run_dir else None
        self._py_logger = logging.getLogger("llm4rec")
        self._py_logger.handlers.clear()
        self._py_logger.setLevel(level)
        self._py_logger.propagate = False

        if console_enabled:
            self._py_logger.addHandler(
                RichHandler(
                    console=self.console,
                    rich_tracebacks=True,
                    show_path=False,
                    markup=True,
                )
            )
        if self._console_log_path is not None:
            self._console_log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(self._console_log_path, encoding="utf-8")
            fh.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            )
            self._py_logger.addHandler(fh)

        if self._jsonl_path is not None:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_path.touch(exist_ok=True)

    def info(self, message: str) -> None:
        self._py_logger.info(message)

    def warning(self, message: str) -> None:
        self._py_logger.warning(message)

    def error(self, message: str) -> None:
        self._py_logger.error(message)

    def log_metrics(
        self,
        metrics: dict[str, float | int | str],
        *,
        stage: str,
        step: int | None = None,
        epoch: float | None = None,
        split: str | None = None,
        wandb_prefix: str | None = None,
    ) -> None:
        """记一批指标到 jsonl + wandb。

        ``wandb_prefix`` 决定这批指标在 wandb 里落到哪个分组，例如
        ``"train"`` → ``train/loss``，``"bias"`` → ``bias/pop_lift@1``。
        默认按 split 推断（train/eval），推不出来就用 stage。
        jsonl 始终写，即使 wandb 不可用也不会丢数据。
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_id": self.experiment_id,
            "stage": stage,
            "split": split,
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
        }
        if self._jsonl_path is not None:
            with self._jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")

        prefix = wandb_prefix or split or stage
        payload = {f"{prefix}/{k}": v for k, v in metrics.items()}
        if epoch is not None:
            payload["progress/epoch"] = epoch
        self.wandb.log(payload, step=step)

    def set_stage(self, stage: str) -> None:
        """切换阶段，在 wandb 上体现为 ``progress/stage_id`` 的跳变。"""
        self.wandb.set_stage(stage)

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        if summary:
            self.wandb.log_summary(summary)
        self.wandb.finish()

    def print_startup_summary(self, lines: list[str]) -> None:
        table = Table(title="Experiment startup", show_header=False, box=None)
        table.add_column("line")
        for line in lines:
            table.add_row(line)
        self.console.print(table)

    def print_stage_summary(
        self,
        rows: list[tuple[str, Any, Any, Any]],
        *,
        title: str = "Stage summary",
    ) -> None:
        table = Table(title=title)
        table.add_column("Metric")
        table.add_column("Best", justify="right")
        table.add_column("Final", justify="right")
        table.add_column("Step", justify="right")
        for metric, best, final, step in rows:
            table.add_row(str(metric), str(best), str(final), str(step))
        self.console.print(table)

    def print_metrics_list(
        self,
        metrics: dict[str, Any] | None,
        *,
        title: str = "Metrics",
        keys: list[str] | None = None,
    ) -> None:
        """Print metrics table to the terminal (startup / final summaries)."""
        table = self._build_metrics_table(metrics, title=title, keys=keys)
        if table is None:
            self.console.print(f"[dim]{title}: (empty)[/dim]")
            return
        self.console.print(table)

    def write_metrics_table(
        self,
        metrics: dict[str, Any] | None,
        *,
        title: str = "Metrics",
        keys: list[str] | None = None,
    ) -> None:
        """Write a metrics table to the process log stream (stdout) + console.log.

        Runner scripts redirect stdout to ``logs/<name>.txt``, so tables land in
        that file. They are not written to the progress TTY.
        """
        table = self._build_metrics_table(metrics, title=title, keys=keys)
        # Primary: stdout (→ logs/smoke.txt etc. when redirected).
        out = Console(
            file=sys.stdout,
            force_terminal=False,
            width=100,
            color_system=None,
            soft_wrap=True,
        )
        if table is None:
            out.print(f"{title}: (empty)")
        else:
            out.print(table)
            out.print("")
        # Secondary: per-run console.log mirror.
        if self._console_log_path is not None:
            self._console_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._console_log_path.open("a", encoding="utf-8") as fh:
                if table is None:
                    fh.write(f"{title}: (empty)\n")
                else:
                    buf = Console(file=fh, force_terminal=False, width=100, color_system=None)
                    buf.print(table)
                    fh.write("\n")

    @staticmethod
    def _build_metrics_table(
        metrics: dict[str, Any] | None,
        *,
        title: str,
        keys: list[str] | None,
    ) -> Table | None:
        if not metrics:
            return None
        table = Table(title=title, show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        items = (
            [(k, metrics[k]) for k in keys if k in metrics]
            if keys is not None
            else list(metrics.items())
        )
        for key, value in items:
            if isinstance(value, float):
                rendered = f"{value:.6g}"
            else:
                rendered = str(value)
            table.add_row(str(key), rendered)
        return table


def build_logger(
    tracking_cfg: dict[str, Any],
    run_dir: Path | None,
    *,
    full_config: dict[str, Any] | None = None,
    experiment_id: str = "",
    is_main_process: bool = True,
) -> ExperimentLogger:
    """构建 logger。传入 ``full_config`` 才会启用 wandb（需要 wandb: 配置段）。"""
    wandb_run: WandbRun | NullRun = NullRun()
    if full_config is not None:
        wandb_run = build_wandb_run(
            full_config,
            run_dir=run_dir,
            experiment_id=experiment_id,
            is_main_process=is_main_process,
        )
    return ExperimentLogger(
        run_dir=run_dir,
        experiment_id=experiment_id,
        console_enabled=bool(tracking_cfg.get("console", True)),
        jsonl_enabled=bool(tracking_cfg.get("jsonl", True)),
        wandb_run=wandb_run,
    )
