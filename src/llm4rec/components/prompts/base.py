"""Prompt builder protocol."""

from __future__ import annotations

from typing import Protocol

from llm4rec.core.schemas import RecommendationExample


class PromptBuilder(Protocol):
    name: str

    def build_messages(
        self,
        *,
        history_titles: list[str],
        candidate_titles: list[str],
        candidate_pop_quantiles: list[float],
        framing: str,
        candidate_genres: list[tuple[str, ...]] | None = None,
        candidate_years: list[int | None] | None = None,
    ) -> list[dict[str, str]]:
        ...
