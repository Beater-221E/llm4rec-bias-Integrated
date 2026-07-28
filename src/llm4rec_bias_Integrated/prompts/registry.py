"""Prompt builder registry."""

from __future__ import annotations

from llm4rec_bias_Integrated.core.registry import Registry
from llm4rec_bias_Integrated.prompts.candidate_choice import CandidateChoicePromptBuilder

PROMPT_REGISTRY: Registry[type] = Registry("prompt")


@PROMPT_REGISTRY.register("candidate_choice")
class _RegisteredCandidateChoice(CandidateChoicePromptBuilder):
    pass


def get_prompt_builder(name: str):
    return PROMPT_REGISTRY.get(name)()
