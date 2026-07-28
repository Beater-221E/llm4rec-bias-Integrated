"""GRPO4Rec workflow package."""

from __future__ import annotations

from typing import Any

__all__ = ["GRPO4RecWorkflow", "task_spec_from_config"]


def __getattr__(name: str) -> Any:
    if name in {"GRPO4RecWorkflow", "task_spec_from_config"}:
        from llm4rec.workflows.grpo4rec import pipeline as _pipeline

        return getattr(_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
