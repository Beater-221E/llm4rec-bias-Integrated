"""Unified CLI entrypoint for llm4rec-bias-Integrated."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from rich.console import Console

from llm4rec import __version__
from llm4rec.core.config import env_compose_overrides, load_config, validate_config
from llm4rec.core.exceptions import ConfigurationError, LabError

console = Console()


def _merge_overrides(cli_overrides: list[str]) -> list[str]:
    """Prepend ``LLM4REC_COMPOSE``; later CLI tokens win on duplicate keys."""
    env_tokens = env_compose_overrides()
    if not env_tokens:
        return list(cli_overrides)
    # Drop env keys that CLI re-specifies so the last explicit token wins cleanly.
    cli_keys = {t.split("=", 1)[0] for t in cli_overrides if "=" in t}
    kept = [t for t in env_tokens if t.split("=", 1)[0] not in cli_keys]
    return kept + list(cli_overrides)


def _add_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style config overrides, e.g. experiment=smoke_test hardware=multi",
    )


def cmd_validate(overrides: list[str]) -> int:
    cfg = load_config(overrides)
    data = validate_config(cfg)
    console.print("[green]Config OK[/green]")
    console.print(
        f"experiment={data['experiment']['name']}  "
        f"dataset={data['dataset']['name']}  "
        f"workflow={data['workflow']['name']}  "
        f"model={data['model']['name']}  "
        f"checkpoint={data['model']['checkpoint']}  "
        f"seed={data['experiment']['seed']}"
    )
    console.print(f"stages={data['training']['stages']}")
    return 0


def cmd_prepare(overrides: list[str]) -> int:
    from llm4rec.cli.prepare import run_prepare

    return run_prepare(overrides)


def cmd_train(overrides: list[str]) -> int:
    # Apply hardware.cuda_visible_devices BEFORE importing torch via train.py.
    load_config(overrides, apply_env=True)
    from llm4rec.cli.train import run_train

    return run_train(overrides)


def cmd_evaluate(overrides: list[str]) -> int:
    from llm4rec.cli.evaluate import run_evaluate

    return run_evaluate(overrides)


def cmd_analyze(overrides: list[str]) -> int:
    from llm4rec.cli.analyze import run_analyze

    return run_analyze(overrides)


def cmd_report(overrides: list[str]) -> int:
    _ = load_config(overrides)
    console.print(
        "[yellow]report[/yellow] is implemented in Phase 9. "
        "Config resolved successfully."
    )
    return 0


def cmd_pipeline(overrides: list[str]) -> int:
    cfg = load_config(overrides)
    data = validate_config(cfg)
    stages = data["training"]["stages"]
    console.print(
        f"[cyan]pipeline[/cyan] experiment={data['experiment']['name']} "
        f"stages={stages}"
    )
    console.print(
        "Full pipeline execution lands with later phases; "
        "Phase 1 validates composition only."
    )
    for stage in stages:
        console.print(f"  · stage '{stage}' — deferred")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm4rec-bias-Integrated",
        description=(
            "llm4rec-bias-Integrated — config-driven experiments for "
            "recommendation bias and reward hacking."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    commands = {
        "validate": "Validate composed experiment config (fail-fast).",
        "prepare": "Download and preprocess datasets.",
        "train": "Run training stage(s) for a workflow.",
        "evaluate": "Evaluate a run directory or checkpoint.",
        "analyze": "Run bias / shortcut probes.",
        "report": "Aggregate runs into tables, plots, and Markdown.",
        "pipeline": "Run the full stage list from an experiment config.",
    }
    for name, help_text in commands.items():
        p = sub.add_parser(name, help=help_text)
        _add_override_args(p)
    return parser


_HANDLERS = {
    "validate": cmd_validate,
    "prepare": cmd_prepare,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "analyze": cmd_analyze,
    "report": cmd_report,
    "pipeline": cmd_pipeline,
}


def main(argv: Sequence[str] | None = None) -> int:
    # When stdout/stderr are redirected to a log file, keep tqdm on one updating line.
    from llm4rec.tracking.inplace_progress import install_inplace_progress

    install_inplace_progress()

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler = _HANDLERS[args.command]
    try:
        return handler(_merge_overrides(list(args.overrides)))
    except ConfigurationError as exc:
        console.print(f"[red]ConfigurationError:[/red] {exc}")
        return 2
    except LabError as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
