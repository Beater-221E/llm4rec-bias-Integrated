"""Workflow registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm4rec.core.registry import Registry
from llm4rec.workflows.base import RecommendationWorkflow

WORKFLOW_REGISTRY: Registry[type[RecommendationWorkflow]] = Registry("workflow")


def register_workflow(
    name: str,
) -> Callable[[type[RecommendationWorkflow]], type[RecommendationWorkflow]]:
    return WORKFLOW_REGISTRY.register(name)


def build_workflow(name: str, **kwargs: Any) -> RecommendationWorkflow:
    # Side-effect imports
    from llm4rec.workflows import grpo4rec as _g  # noqa: F401
    from llm4rec.workflows import minionerec as _mini  # noqa: F401
    from llm4rec.workflows import mllm4rec as _m  # noqa: F401

    cls = WORKFLOW_REGISTRY.get(name)
    return cls(**kwargs)
