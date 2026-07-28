"""``llm4rec-bias-Integrated analyze`` — bias / shortcut probes (Phase 6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.pretty import Pretty

from llm4rec.core.config import load_config, validate_config
from llm4rec.core.exceptions import ConfigurationError
from llm4rec.core.paths import project_root
from llm4rec.components.evaluation.probes.runner import run_probes

console = Console()


def _resolve_run_dir(cfg: dict[str, Any], overrides: list[str]) -> Path:
    raw = None
    for token in overrides:
        if token.startswith("run_dir="):
            raw = token.split("=", 1)[1]
    if raw is None:
        raw = (cfg.get("evaluation") or {}).get("run_dir") or cfg.get("run_dir")
    if not raw:
        raise ConfigurationError(
            "analyze requires run_dir=... (path to a finished training run)"
        )
    path = Path(raw)
    return path if path.is_absolute() else project_root() / path


def _override_value(overrides: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for token in overrides:
        if token.startswith(prefix):
            return token.split("=", 1)[1]
    return None


def run_analyze(overrides: list[str]) -> int:
    cfg_omega = load_config(overrides)
    cfg = validate_config(cfg_omega)
    run_dir = _resolve_run_dir(cfg, overrides)
    if not run_dir.is_dir():
        raise ConfigurationError(f"run_dir does not exist: {run_dir}")

    checkpoint_stage = _override_value(overrides, "checkpoint_stage") or (
        (cfg.get("bias") or {}).get("checkpoint_stage")
    )
    adapter_path = _override_value(overrides, "adapter_path")
    sft_adapter_path = _override_value(overrides, "sft_adapter_path")
    split = str(
        _override_value(overrides, "evaluation.split")
        or (cfg.get("evaluation") or {}).get("split")
        or "test"
    )

    summary = run_probes(
        cfg=cfg,
        run_dir=run_dir,
        checkpoint_stage=checkpoint_stage,
        adapter_path=adapter_path,
        sft_adapter_path=sft_adapter_path,
        split=split,
    )
    console.print("[green]Analyze finished[/green]")
    console.print(Pretty(summary))
    return 0
