"""Compatibility CLI shim — re-exports and runs main when invoked as __main__."""
from llm4rec.workflows.mllm4rec._stack.cli import *  # noqa: F403
from llm4rec.workflows.mllm4rec._stack.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
