#!/usr/bin/env python
"""Unified training entrypoint for the LLM4Rec Research Framework.

Examples::

    python train.py workflow=grpo4rec dataset=ml1m model=qwen25_3b reward=bias_aware evaluation=full_bias
    python train.py workflow=minionerec experiment=smoke_sid
    python train.py workflow=mllm4rec

Compose selectors (Hydra-style) are merged via ``llm4rec.core.config.load_config``.
Hardware / scale still accept ``LLM4REC_COMPOSE`` or ``hardware=`` / ``scale=``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without install
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main(argv: list[str] | None = None) -> int:
    from llm4rec.cli.main import main as cli_main

    args = list(argv if argv is not None else sys.argv[1:])
    # Default command is train when first token looks like an override
    if not args:
        args = ["train"]
    elif "=" in args[0] or args[0] not in {
        "validate",
        "prepare",
        "train",
        "evaluate",
        "analyze",
        "report",
        "pipeline",
    }:
        args = ["train", *args]
    return cli_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
