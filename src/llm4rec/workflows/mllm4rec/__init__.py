"""MLLM4Rec workflow package."""

from __future__ import annotations

from typing import Any

__all__ = ["MLLM4RecWorkflow"]


def __getattr__(name: str) -> Any:
    if name == "MLLM4RecWorkflow":
        from llm4rec.workflows.mllm4rec.pipeline import MLLM4RecWorkflow

        return MLLM4RecWorkflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
