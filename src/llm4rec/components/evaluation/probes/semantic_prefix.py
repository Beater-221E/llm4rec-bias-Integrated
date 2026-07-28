"""Semantic-prefix probe — deferred to Phase 7 (SID path)."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llm4rec.core.exceptions import ConfigurationError
from llm4rec.core.schemas import ProbeResult, RecommendationExample
from llm4rec.components.evaluation.probes.base import BiasProbe
from llm4rec.components.evaluation.probes.registry import register_probe


@register_probe("semantic_prefix")
class SemanticPrefixProbe(BiasProbe):
    name = "semantic_prefix"

    def __init__(self, **_: Any) -> None:
        raise ConfigurationError(
            "probe 'semantic_prefix' requires the MiniOneRec SID path (Phase 7). "
            "Use letter-route probes: popularity, position, framing, recency, permutation."
        )

    def run(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        examples: list[RecommendationExample],
        *,
        device: Any,
        cfg: dict[str, Any] | None = None,
    ) -> ProbeResult:
        raise ConfigurationError("semantic_prefix is Phase 7")
