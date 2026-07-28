"""Candidate list construction (negatives + target placement)."""

from __future__ import annotations

import random
from typing import Literal

from llm4rec_bias_Integrated.core.exceptions import ConfigurationError
from llm4rec_bias_Integrated.datasets.sampling.base import NegativeSampler

TargetPosition = Literal["random", "first", "last", "middle"]


def place_target(
    target_item_id: str,
    negatives: list[str],
    *,
    candidate_size: int,
    target_position: TargetPosition | str,
    rng: random.Random,
) -> tuple[list[str], int]:
    """Insert target into negatives to form a candidate list of size C."""
    if len(negatives) < candidate_size - 1:
        raise ConfigurationError(
            f"Need {candidate_size - 1} negatives, got {len(negatives)}"
        )
    negs = negatives[: candidate_size - 1]
    if target_position == "first":
        pos = 0
    elif target_position == "last":
        pos = candidate_size - 1
    elif target_position == "middle":
        pos = candidate_size // 2
    elif target_position == "random":
        pos = rng.randrange(candidate_size)
    else:
        raise ConfigurationError(f"Unknown target_position '{target_position}'")
    candidates = negs[:pos] + [target_item_id] + negs[pos:]
    return candidates, pos


def build_candidate_list(
    *,
    target_item_id: str,
    exclude: set[str],
    sampler: NegativeSampler,
    candidate_size: int,
    target_position: str,
    rng: random.Random,
) -> tuple[list[str], int]:
    negs = sampler.sample(k=candidate_size - 1, exclude=exclude | {target_item_id}, rng=rng)
    return place_target(
        target_item_id,
        negs,
        candidate_size=candidate_size,
        target_position=target_position,
        rng=rng,
    )
