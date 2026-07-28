"""MLLM4Rec workflow — Phase 3: text-only candidate choice shares SFT path.

Full multimodal content provider lands in Phase 8. For now this workflow
reuses the GRPO4Rec candidate-choice SFT composition.
"""

from __future__ import annotations

from typing import Any

from llm4rec_bias_Integrated.workflows.grpo4rec import GRPO4RecWorkflow
from llm4rec_bias_Integrated.workflows.registry import register_workflow


@register_workflow("mllm4rec")
class MLLM4RecWorkflow(GRPO4RecWorkflow):
    """Text-only v1 of MLLM4Rec (discriminative candidate choice)."""

    name = "mllm4rec"

    def required_stages(self) -> list[str]:
        return [
            "prepare_data",
            "build_candidate_sets",
            "build_text_or_multimodal_prompts",
            "sft",
            "optional_rl",
            "candidate_ranking_evaluation",
            "counterfactual_probes",
            "report",
        ]

    def build_probes(self, context) -> list[Any]:
        return []
