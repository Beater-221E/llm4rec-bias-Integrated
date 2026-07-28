"""Datasets package."""

from llm4rec.components.dataset._impl.registry import build_dataset, register_dataset

__all__ = ["build_dataset", "register_dataset"]
