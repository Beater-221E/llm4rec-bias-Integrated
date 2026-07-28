"""Workflow smoke tests — interface + registration (no training)."""

from __future__ import annotations

from llm4rec.workflows.base import BaseWorkflow
from llm4rec.workflows.grpo4rec import GRPO4RecWorkflow
from llm4rec.workflows.minionerec import MiniOneRecWorkflow
from llm4rec.workflows.mllm4rec import MLLM4RecWorkflow
from llm4rec.workflows.registry import WORKFLOW_REGISTRY, build_workflow, get_workflow_class


def test_workflows_registered():
    # Force import side effects
    get_workflow_class("grpo4rec")
    assert WORKFLOW_REGISTRY.contains("grpo4rec")
    assert WORKFLOW_REGISTRY.contains("minionerec")
    assert WORKFLOW_REGISTRY.contains("mllm4rec")


def test_build_workflow_instances():
    g = build_workflow("grpo4rec")
    m = build_workflow("minionerec")
    l = build_workflow("mllm4rec")
    assert isinstance(g, GRPO4RecWorkflow)
    assert isinstance(m, MiniOneRecWorkflow)
    assert isinstance(l, MLLM4RecWorkflow)
    assert isinstance(g, BaseWorkflow)
    assert g.name == "grpo4rec"
    assert m.name == "minionerec"
    assert l.name == "mllm4rec"


def test_workflow_interface_methods():
    for cls in (GRPO4RecWorkflow, MiniOneRecWorkflow, MLLM4RecWorkflow):
        for method in ("prepare_data", "build_model", "train", "evaluate", "inference"):
            assert callable(getattr(cls, method))


def test_workflows_are_independent_classes():
    assert not issubclass(MLLM4RecWorkflow, GRPO4RecWorkflow)
    assert not issubclass(MiniOneRecWorkflow, GRPO4RecWorkflow)
