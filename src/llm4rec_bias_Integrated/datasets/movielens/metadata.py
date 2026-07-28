"""Genre / title normalization helpers for MovieLens."""

from __future__ import annotations

import re

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")

# ML-100K genre index order (u.item columns 5..23)
ML100K_GENRES: tuple[str, ...] = (
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)


def parse_title_year(title: str) -> tuple[str, int | None]:
    """Split ``Toy Story (1995)`` into title + year."""
    match = _YEAR_RE.search(title.strip())
    if not match:
        return title.strip(), None
    year = int(match.group(1))
    clean = title[: match.start()].strip()
    return clean or title.strip(), year


def genres_from_ml100k_flags(flags: list[str]) -> tuple[str, ...]:
    genres: list[str] = []
    for idx, flag in enumerate(flags[: len(ML100K_GENRES)]):
        if flag.strip() == "1":
            name = ML100K_GENRES[idx]
            if name != "unknown":
                genres.append(name)
    return tuple(genres)


def genres_from_pipe_string(raw: str) -> tuple[str, ...]:
    parts = [g.strip() for g in raw.split("|") if g.strip()]
    return tuple(parts)
