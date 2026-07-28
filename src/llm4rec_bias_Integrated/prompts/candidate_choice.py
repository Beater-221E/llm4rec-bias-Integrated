"""Candidate-choice (letter) prompts and strict parser.

Ported from llm4rec-bias ``prompts.py`` with framing extensions.
Reward and evaluation MUST share ``parse_choice``.
"""

from __future__ import annotations

import string

from llm4rec_bias_Integrated.datasets.transforms.framing import format_candidate_line

LETTERS = string.ascii_uppercase

SYSTEM_PROMPT = (
    "You are a movie recommender. Given a user's watch history and a list of "
    "candidate movies, pick the single candidate the user is most likely to watch next. "
    "Answer with only the letter of your choice."
)


def render_candidates(
    titles: list[str],
    pop_quantiles: list[float],
    framing: str,
    *,
    genres: list[tuple[str, ...]] | None = None,
    years: list[int | None] | None = None,
) -> str:
    lines = []
    for i, title in enumerate(titles):
        g = genres[i] if genres is not None else ()
        y = years[i] if years is not None else None
        lines.append(
            format_candidate_line(
                LETTERS[i],
                title,
                framing=framing,
                pop_quantile=pop_quantiles[i],
                genres=g,
                release_year=y,
            )
        )
    return "\n".join(lines)


def build_candidate_choice_messages(
    history_titles: list[str],
    candidate_titles: list[str],
    pop_quantiles: list[float],
    framing: str = "neutral",
    *,
    candidate_genres: list[tuple[str, ...]] | None = None,
    candidate_years: list[int | None] | None = None,
) -> list[dict[str, str]]:
    hist = "\n".join(f"- {t}" for t in history_titles)
    cands = render_candidates(
        candidate_titles,
        pop_quantiles,
        framing,
        genres=candidate_genres,
        years=candidate_years,
    )
    user = (
        f"Movies this user watched recently (oldest to newest):\n{hist}\n\n"
        f"Candidates:\n{cands}\n\n"
        f"Which candidate will the user watch next? Answer with only the letter."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_choice(completion: str, num_candidates: int) -> int | None:
    """Extract chosen candidate index, or None if invalid.

    Accepts ``B``, ``B.``, ``Answer: B``, ``B. Fargo (1996)`` on the first
    non-empty line. Rejects word-prefix false positives (e.g. ``Based on...``).
    """
    text = completion.strip()
    if not text:
        return None
    first = text.splitlines()[0].strip()
    if first.lower().startswith("answer:"):
        first = first[len("answer:") :].strip()
    if not first:
        return None
    ch = first[0].upper()
    if len(first) > 1 and first[1] not in ".):, ":
        return None
    if ch in LETTERS[:num_candidates]:
        return LETTERS.index(ch)
    return None


class CandidateChoicePromptBuilder:
    name = "candidate_choice"

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
        return build_candidate_choice_messages(
            history_titles,
            candidate_titles,
            candidate_pop_quantiles,
            framing,
            candidate_genres=candidate_genres,
            candidate_years=candidate_years,
        )
