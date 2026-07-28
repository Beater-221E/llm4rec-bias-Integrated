"""Text framing variants for candidate rendering."""

from __future__ import annotations

from typing import Callable

FRAMING_VARIANTS = (
    "neutral",
    "evaluative",
    "popularity_marked",
    "concise",
    "verbose",
    "paraphrased",
)


def format_candidate_line(
    letter: str,
    title: str,
    *,
    framing: str,
    pop_quantile: float,
    genres: tuple[str, ...] = (),
    release_year: int | None = None,
) -> str:
    """Render one candidate line under a framing policy."""
    if framing == "neutral":
        return f"{letter}. {title}"
    if framing in {"evaluative", "popularity_marked"}:
        if pop_quantile >= 0.75:
            suffix = " (popular hit)"
        elif pop_quantile <= 0.25:
            suffix = " (rarely watched)"
        else:
            suffix = ""
        return f"{letter}. {title}{suffix}"
    if framing == "concise":
        short = title if len(title) <= 40 else title[:37] + "..."
        return f"{letter}. {short}"
    if framing == "verbose":
        bits = [title]
        if release_year is not None:
            bits.append(f"released {release_year}")
        if genres:
            bits.append("genres: " + ", ".join(genres[:3]))
        bits.append(f"popularity quantile {pop_quantile:.2f}")
        return f"{letter}. " + "; ".join(bits)
    if framing == "paraphrased":
        return f"{letter}. Next option: {title}"
    # unknown → neutral
    return f"{letter}. {title}"


def framing_gap_label(a: str, b: str) -> str:
    return f"{a}_vs_{b}"
