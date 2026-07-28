"""Shared dataset processors (filter / split / sequence build)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from llm4rec.components.dataset.base import DatasetBundle
from llm4rec.core.schemas import Interaction


class DatasetProcessor:
    """Stateless transforms over interactions → sequences / metadata."""

    @staticmethod
    def unique_ids(interactions: Iterable[Interaction]) -> tuple[list[str], list[str]]:
        users: set[str] = set()
        items: set[str] = set()
        for row in interactions:
            users.add(row.user_id)
            items.add(row.item_id)
        return sorted(users), sorted(items)

    @staticmethod
    def build_sequences(
        interactions: Iterable[Interaction],
        *,
        max_len: int | None = None,
    ) -> dict[str, list[str]]:
        seqs: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for row in interactions:
            seqs[row.user_id].append((int(row.timestamp), row.item_id))
        out: dict[str, list[str]] = {}
        for user, events in seqs.items():
            events.sort(key=lambda x: x[0])
            items = [item for _, item in events]
            if max_len is not None and max_len > 0:
                items = items[-max_len:]
            out[user] = items
        return out

    @staticmethod
    def filter_min_interactions(
        interactions: list[Interaction],
        *,
        min_user: int = 5,
        min_item: int = 0,
    ) -> list[Interaction]:
        user_counts: dict[str, int] = defaultdict(int)
        item_counts: dict[str, int] = defaultdict(int)
        for row in interactions:
            user_counts[row.user_id] += 1
            item_counts[row.item_id] += 1
        keep_users = {u for u, c in user_counts.items() if c >= min_user}
        keep_items = (
            {i for i, c in item_counts.items() if c >= min_item}
            if min_item > 0
            else None
        )
        filtered = [
            row
            for row in interactions
            if row.user_id in keep_users
            and (keep_items is None or row.item_id in keep_items)
        ]
        return filtered

    @staticmethod
    def enrich_bundle(
        bundle: DatasetBundle,
        *,
        extras: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetBundle:
        new_extras = dict(bundle.extras)
        if extras:
            new_extras.update(extras)
        new_meta = dict(bundle.metadata)
        if metadata:
            new_meta.update(metadata)
        return DatasetBundle(
            name=bundle.name,
            interactions=bundle.interactions,
            users=bundle.users,
            items=bundle.items,
            sequences=dict(bundle.sequences),
            metadata=new_meta,
            extras=new_extras,
        )
