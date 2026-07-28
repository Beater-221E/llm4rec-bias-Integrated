"""Semantic-ID generative-retrieval prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm4rec_bias_Integrated.semantic_ids.table import SidTable

SYSTEM_PROMPT = (
    "You are a movie recommender. Every movie has a semantic ID made of codes "
    "like <s0_12><s1_45><s2_7><s3_0>; similar movies share leading codes. "
    "Given a user's watch history, respond with only the semantic ID of the "
    "movie they will watch next."
)


def build_sid_messages(
    *,
    history_item_ids: list[str],
    table: "SidTable",
    titles: dict[str, str] | None = None,
    with_titles: bool = True,
) -> list[dict[str, str]]:
    if with_titles and titles is not None:
        lines = [f"- {titles.get(i, i)} {table.sid(i)}" for i in history_item_ids]
    else:
        lines = [f"- {table.sid(i)}" for i in history_item_ids]
    user_msg = (
        "Movies this user watched recently (oldest to newest):\n"
        + "\n".join(lines)
        + "\n\nSemantic ID of the next movie:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
