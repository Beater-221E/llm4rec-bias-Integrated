"""Dataset adapter registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm4rec.core.registry import Registry
from llm4rec.components.dataset._impl.base import DatasetAdapter

DATASET_REGISTRY: Registry[type[DatasetAdapter]] = Registry("dataset")


def register_dataset(name: str) -> Callable[[type[DatasetAdapter]], type[DatasetAdapter]]:
    return DATASET_REGISTRY.register(name)


def get_dataset_class(name: str) -> type[DatasetAdapter]:
    return DATASET_REGISTRY.get(name)


def build_dataset(name: str, **kwargs: Any) -> DatasetAdapter:
    """Instantiate a registered dataset adapter."""
    # Import side-effect registrations
    from llm4rec.components.dataset._impl import movielens as _movielens  # noqa: F401

    cls = get_dataset_class(name)
    return cls(**kwargs)
