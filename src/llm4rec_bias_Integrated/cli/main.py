"""Compatibility CLI shim — re-exports and runs main when invoked as __main__."""
from llm4rec.cli.main import *  # noqa: F403
from llm4rec.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
