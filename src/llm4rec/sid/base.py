"""Shared SID helpers: metrics and small utilities (no heavy models)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass
class CollisionMetrics:
    """Explicit SID collision diagnostics.

    Distinguishes catalog duplicate metadata (identical embedding texts) from
    genuine quantization collisions when ``duplicate_item_ids`` is provided.
    """

    raw_collision_rate: float
    post_resolution_collision_rate: float
    num_collision_groups: int
    max_collision_group_size: int
    duplicate_item_collision_rate: float
    quantization_collision_rate: float
    n_items: int
    n_unique_sids: int

    def to_dict(self) -> dict:
        return asdict(self)


def codes_as_tuples(codes: np.ndarray) -> list[tuple[int, ...]]:
    return [tuple(int(c) for c in row) for row in codes]


def collision_rate(codes: np.ndarray) -> float:
    """Fraction of items that share their SID with at least one other item."""
    n = int(codes.shape[0])
    if n == 0:
        return 0.0
    unique = len(set(codes_as_tuples(codes)))
    return float((n - unique) / n)


def collision_groups(codes: np.ndarray) -> list[list[int]]:
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i, row in enumerate(codes):
        groups[tuple(int(c) for c in row)].append(i)
    return [g for g in groups.values() if len(g) > 1]


def compute_collision_metrics(
    codes: np.ndarray,
    *,
    raw_codes: np.ndarray | None = None,
    duplicate_groups: Sequence[Sequence[int]] | None = None,
) -> CollisionMetrics:
    """Compute collision metrics for a SID assignment.

    ``duplicate_groups`` lists index groups that are known catalog duplicates
    (e.g. identical item text). Those collisions are attributed to
    ``duplicate_item_collision_rate``; the remainder is
    ``quantization_collision_rate``.
    """
    n = int(codes.shape[0])
    tuples = codes_as_tuples(codes)
    unique = len(set(tuples))
    groups = collision_groups(codes)
    post = float((n - unique) / n) if n else 0.0
    raw = collision_rate(raw_codes) if raw_codes is not None else post

    dup_indices: set[int] = set()
    if duplicate_groups:
        for g in duplicate_groups:
            if len(g) > 1:
                dup_indices.update(int(i) for i in g)

    colliding = {i for g in groups for i in g}
    dup_colliding = colliding & dup_indices if dup_indices else set()
    # Items that collide only because of quantization (not known duplicates)
    quant_colliding = colliding - dup_indices if dup_indices else colliding

    return CollisionMetrics(
        raw_collision_rate=raw,
        post_resolution_collision_rate=post,
        num_collision_groups=len(groups),
        max_collision_group_size=max((len(g) for g in groups), default=0),
        duplicate_item_collision_rate=float(len(dup_colliding) / n) if n else 0.0,
        quantization_collision_rate=float(len(quant_colliding) / n) if n else 0.0,
        n_items=n,
        n_unique_sids=unique,
    )


def find_duplicate_text_groups(texts: Sequence[str]) -> list[list[int]]:
    """Group item indices that share identical text (catalog duplicates)."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, text in enumerate(texts):
        buckets[str(text)].append(i)
    return [g for g in buckets.values() if len(g) > 1]
