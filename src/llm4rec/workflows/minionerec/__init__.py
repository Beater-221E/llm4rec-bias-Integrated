"""MiniOneRec workflow package."""

from __future__ import annotations

from typing import Any

__all__ = ["MiniOneRecWorkflow"]


def __getattr__(name: str) -> Any:
    if name == "MiniOneRecWorkflow":
        from llm4rec.workflows.minionerec.pipeline import MiniOneRecWorkflow

        return MiniOneRecWorkflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
