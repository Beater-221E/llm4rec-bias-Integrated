"""Workflows package."""

from llm4rec.workflows.base import BaseWorkflow, RecommendationWorkflow
from llm4rec.workflows.registry import build_workflow, register_workflow

__all__ = [
    "BaseWorkflow",
    "RecommendationWorkflow",
    "build_workflow",
    "register_workflow",
]
