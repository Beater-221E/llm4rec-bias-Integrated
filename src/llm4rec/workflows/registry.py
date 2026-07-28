"""Workflow registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm4rec.core.registry import Registry
from llm4rec.workflows.base import BaseWorkflow, RecommendationWorkflow

WORKFLOW_REGISTRY: Registry[type] = Registry("workflow")


def register_workflow(name: str) -> Callable[[type], type]:
    return WORKFLOW_REGISTRY.register(name)


def get_workflow_class(name: str) -> type:
    # Import pipeline modules (not just packages) so @register_workflow runs.
    from llm4rec.workflows.grpo4rec import pipeline as _g  # noqa: F401
    from llm4rec.workflows.minionerec import pipeline as _m  # noqa: F401
    from llm4rec.workflows.mllm4rec import pipeline as _l  # noqa: F401

    return WORKFLOW_REGISTRY.get(name)


def build_workflow(name: str, **kwargs: Any) -> RecommendationWorkflow | BaseWorkflow:
    cls = get_workflow_class(name)
    return cls(**kwargs)


__all__ = [
    "WORKFLOW_REGISTRY",
    "register_workflow",
    "get_workflow_class",
    "build_workflow",
]
