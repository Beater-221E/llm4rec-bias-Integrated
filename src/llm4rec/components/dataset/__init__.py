"""Dataset layer: DatasetBuilder → DatasetBundle."""

from __future__ import annotations

from typing import Any

__all__ = [
    "DatasetBuilder",
    "DatasetBundle",
    "DatasetProcessor",
    "InteractionSchema",
    "MovieLens100KBuilder",
    "MovieLens1MBuilder",
    "build_movielens_bundle",
]


def __getattr__(name: str) -> Any:
    if name in {"DatasetBuilder", "DatasetBundle"}:
        from llm4rec.components.dataset import base as _base

        return getattr(_base, name)
    if name == "InteractionSchema":
        from llm4rec.components.dataset.schema import InteractionSchema

        return InteractionSchema
    if name == "DatasetProcessor":
        from llm4rec.components.dataset.processor import DatasetProcessor

        return DatasetProcessor
    if name in {
        "MovieLens100KBuilder",
        "MovieLens1MBuilder",
        "build_movielens_bundle",
    }:
        from llm4rec.components.dataset import movielens as _ml

        return getattr(_ml, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
