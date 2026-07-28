"""Prompts package."""

from __future__ import annotations

from typing import Any

__all__ = [
    "LETTERS",
    "build_candidate_choice_messages",
    "parse_choice",
]


def __getattr__(name: str) -> Any:
    from llm4rec.components.prompts import candidate_choice as _cc

    if name in __all__:
        return getattr(_cc, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
