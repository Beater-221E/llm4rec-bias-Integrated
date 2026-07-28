"""Negative sampling strategies."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from llm4rec.core.exceptions import ConfigurationError
from llm4rec.core.registry import Registry

SAMPLER_REGISTRY: Registry[type["NegativeSampler"]] = Registry("negative_sampler")


class NegativeSampler(ABC):
    name: str

    @abstractmethod
    def sample(
        self,
        *,
        k: int,
        exclude: set[str],
        rng: random.Random,
    ) -> list[str]:
        """Sample ``k`` item IDs not in ``exclude``."""


def register_sampler(name: str):
    return SAMPLER_REGISTRY.register(name)


def get_sampler(name: str, **kwargs) -> NegativeSampler:
    # Side-effect imports
    from llm4rec.components.dataset._impl.sampling import (  # noqa: F401
        exposure_matched,
        hard_negative,
        popularity,
        uniform,
    )

    key = "pop" if name == "pop" else name
    if key == "pop":
        key = "popularity"
    cls = SAMPLER_REGISTRY.get(key)
    return cls(**kwargs)


@register_sampler("uniform")
class UniformNegativeSampler(NegativeSampler):
    name = "uniform"

    def __init__(self, item_ids: Sequence[str]) -> None:
        self.item_ids = list(item_ids)
        if not self.item_ids:
            raise ConfigurationError("UniformNegativeSampler requires a non-empty pool")

    def sample(
        self,
        *,
        k: int,
        exclude: set[str],
        rng: random.Random,
    ) -> list[str]:
        pool = [i for i in self.item_ids if i not in exclude]
        if len(pool) < k:
            raise ConfigurationError(
                f"Need {k} negatives but only {len(pool)} items available"
            )
        return rng.sample(pool, k)


@register_sampler("popularity")
class PopularityNegativeSampler(NegativeSampler):
    name = "popularity"

    def __init__(self, item_ids: Sequence[str], counts: dict[str, int]) -> None:
        self.item_ids = [i for i in item_ids if i in counts]
        if not self.item_ids:
            raise ConfigurationError("PopularityNegativeSampler requires counted items")
        weights = np.array([float(counts[i]) for i in self.item_ids], dtype=np.float64)
        total = weights.sum()
        if total <= 0:
            raise ConfigurationError("Popularity weights sum to zero")
        self.probs = weights / total

    def sample(
        self,
        *,
        k: int,
        exclude: set[str],
        rng: random.Random,
    ) -> list[str]:
        negs: list[str] = []
        seen = set(exclude)
        # Seeded numpy RNG derived from python RNG for reproducibility
        np_rng = np.random.default_rng(rng.randrange(2**31))
        guard = 0
        while len(negs) < k:
            guard += 1
            if guard > 10_000:
                raise ConfigurationError("Failed to sample enough popularity negatives")
            batch = np_rng.choice(self.item_ids, size=min(k * 4, len(self.item_ids)), p=self.probs)
            for item in batch:
                item_id = str(item)
                if item_id in seen:
                    continue
                seen.add(item_id)
                negs.append(item_id)
                if len(negs) == k:
                    break
        return negs


@register_sampler("hard_negative")
class HardNegativeSampler(NegativeSampler):
    """Phase-2 placeholder: falls back to popularity sampling.

    True hard negatives (model-based / CF) land with trainers.
    """

    name = "hard_negative"

    def __init__(self, item_ids: Sequence[str], counts: dict[str, int]) -> None:
        self._inner = PopularityNegativeSampler(item_ids, counts)

    def sample(
        self,
        *,
        k: int,
        exclude: set[str],
        rng: random.Random,
    ) -> list[str]:
        return self._inner.sample(k=k, exclude=exclude, rng=rng)


@register_sampler("exposure_matched")
class ExposureMatchedNegativeSampler(NegativeSampler):
    """Sample negatives with popularity close to the target's quantile.

    ``compatibility approximation`` of exposure-matched negatives.
    """

    name = "exposure_matched"

    def __init__(
        self,
        item_ids: Sequence[str],
        quantiles: dict[str, float],
        *,
        target_quantile: float = 0.5,
        bandwidth: float = 0.1,
    ) -> None:
        self.item_ids = list(item_ids)
        self.quantiles = quantiles
        self.target_quantile = target_quantile
        self.bandwidth = bandwidth

    def sample(
        self,
        *,
        k: int,
        exclude: set[str],
        rng: random.Random,
    ) -> list[str]:
        band = [
            i
            for i in self.item_ids
            if i not in exclude
            and abs(self.quantiles.get(i, 0.5) - self.target_quantile) <= self.bandwidth
        ]
        if len(band) < k:
            # widen pool if band too small
            band = [i for i in self.item_ids if i not in exclude]
        if len(band) < k:
            raise ConfigurationError(
                f"Need {k} exposure-matched negatives but only {len(band)} available"
            )
        return rng.sample(band, k)
