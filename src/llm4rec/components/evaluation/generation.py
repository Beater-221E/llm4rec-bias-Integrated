"""Generation metrics: SID validity, semantic collision, generation accuracy."""

from __future__ import annotations

from typing import Any, Iterable, Sequence


class GenerationMetrics:
    """Metrics for generative recommendation (SID / free-form)."""

    @staticmethod
    def sid_validity(parsed_ok: Iterable[bool]) -> dict[str, float]:
        flags = list(parsed_ok)
        n = max(len(flags), 1)
        valid = sum(1 for x in flags if x)
        return {
            "sid_validity": valid / n,
            "sid_invalid_rate": 1.0 - valid / n,
            "n": float(len(flags)),
        }

    @staticmethod
    def semantic_collision(
        predicted_sids: Sequence[tuple[Any, ...]],
        *,
        sid_to_items: dict[tuple[Any, ...], list[Any]] | None = None,
    ) -> dict[str, float]:
        """Fraction of predicted SIDs that map to >1 item (collision)."""
        if not predicted_sids:
            return {"semantic_collision_rate": 0.0, "n": 0.0}
        if sid_to_items is None:
            from collections import Counter

            counts = Counter(predicted_sids)
            collisions = sum(1 for s in predicted_sids if counts[s] > 1)
            return {
                "semantic_collision_rate": collisions / len(predicted_sids),
                "n": float(len(predicted_sids)),
            }
        collisions = 0
        for sid in predicted_sids:
            items = sid_to_items.get(tuple(sid), [])
            if len(items) > 1:
                collisions += 1
        return {
            "semantic_collision_rate": collisions / len(predicted_sids),
            "n": float(len(predicted_sids)),
        }

    @staticmethod
    def generation_accuracy(
        exact_matches: Iterable[bool],
    ) -> dict[str, float]:
        flags = list(exact_matches)
        n = max(len(flags), 1)
        return {
            "generation_accuracy": sum(1 for x in flags if x) / n,
            "n": float(len(flags)),
        }


def evaluate_sid_checkpoint(*args: Any, **kwargs: Any):
    """Lazy wrapper to avoid circular imports with MiniOneRec."""
    from llm4rec.components.evaluation._impl.sid import (
        evaluate_sid_checkpoint as _impl,
    )

    return _impl(*args, **kwargs)


__all__ = ["GenerationMetrics", "evaluate_sid_checkpoint"]
