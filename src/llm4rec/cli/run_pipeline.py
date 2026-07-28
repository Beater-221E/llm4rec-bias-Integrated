"""``python -m llm4rec.cli.run_pipeline`` entry."""

from __future__ import annotations

import sys

from llm4rec.cli.main import main


def run() -> None:
    # Default to pipeline subcommand when invoked as module.
    argv = list(sys.argv[1:])
    if not argv or argv[0].startswith("-") or "=" in argv[0]:
        argv = ["pipeline", *argv]
    elif argv[0] not in {
        "validate",
        "prepare",
        "train",
        "evaluate",
        "analyze",
        "report",
        "pipeline",
    }:
        argv = ["pipeline", *argv]
    raise SystemExit(main(argv))


if __name__ == "__main__":
    run()
