"""Bias probe base interface (Phase 6)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llm4rec.core.schemas import ProbeResult, RecommendationExample


class BiasProbe(ABC):
    """Counterfactual / observational bias probe on letter-route examples."""

    name: str

    @abstractmethod
    def run(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        examples: list[RecommendationExample],
        *,
        device: Any,
        cfg: dict[str, Any] | None = None,
    ) -> ProbeResult:
        """Score examples (and counterfactuals) and return metrics."""
