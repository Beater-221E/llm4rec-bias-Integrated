"""MovieLens shared types and statistics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from llm4rec.core.schemas import Interaction


@dataclass(frozen=True)
class ItemMetadata:
    """Catalog metadata for one item (string IDs throughout)."""

    item_id: str
    title: str
    genres: tuple[str, ...] = ()
    release_year: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def chronological_sequences(
    interactions: list[Interaction],
    *,
    dedupe_consecutive: bool = True,
) -> dict[str, list[Interaction]]:
    """Group interactions by user, sorted by timestamp then item_id."""
    by_user: dict[str, list[Interaction]] = defaultdict(list)
    for ix in interactions:
        by_user[ix.user_id].append(ix)
    sequences: dict[str, list[Interaction]] = {}
    for user_id, rows in by_user.items():
        rows = sorted(rows, key=lambda x: (x.timestamp, x.item_id))
        if dedupe_consecutive:
            deduped: list[Interaction] = []
            for row in rows:
                if not deduped or deduped[-1].item_id != row.item_id:
                    deduped.append(row)
            rows = deduped
        sequences[user_id] = rows
    return sequences


def filter_sequences_by_length(
    sequences: dict[str, list[Interaction]],
    min_length: int,
) -> dict[str, list[Interaction]]:
    return {u: s for u, s in sequences.items() if len(s) >= min_length}


def popularity_from_train_region(
    sequences: dict[str, list[Interaction]],
    *,
    holdout: int = 2,
) -> tuple[dict[str, int], dict[str, float]]:
    """Counts / quantiles on interactions before the last ``holdout`` per user.

    Using only the training region avoids leaking validation/test into popularity.
    """
    counts: dict[str, int] = defaultdict(int)
    for seq in sequences.values():
        for ix in seq[:-holdout] if holdout > 0 else seq:
            counts[ix.item_id] += 1
    items = sorted(counts)
    ordered = sorted(items, key=lambda i: (counts[i], i))
    n = len(ordered)
    quantiles = {
        item_id: rank / max(n - 1, 1) for rank, item_id in enumerate(ordered)
    }
    return dict(counts), quantiles


def popularity_summary(
    counts: dict[str, int],
    quantiles: dict[str, float],
) -> dict[str, Any]:
    if not counts:
        return {
            "n_items": 0,
            "total_interactions": 0,
            "mean_count": 0.0,
            "median_count": 0.0,
            "max_count": 0,
            "min_count": 0,
        }
    values = sorted(counts.values())
    mid = len(values) // 2
    median = (
        values[mid]
        if len(values) % 2 == 1
        else 0.5 * (values[mid - 1] + values[mid])
    )
    return {
        "n_items": len(counts),
        "total_interactions": int(sum(values)),
        "mean_count": float(sum(values) / len(values)),
        "median_count": float(median),
        "max_count": int(values[-1]),
        "min_count": int(values[0]),
        "quantile_mean": float(sum(quantiles.values()) / max(len(quantiles), 1)),
    }
