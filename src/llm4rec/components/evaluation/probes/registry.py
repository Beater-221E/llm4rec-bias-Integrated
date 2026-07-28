"""Bias probe registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm4rec.core.registry import Registry
from llm4rec.components.evaluation.probes.base import BiasProbe

PROBE_REGISTRY: Registry[type] = Registry("probe")


def register_probe(name: str) -> Callable[[type], type]:
    return PROBE_REGISTRY.register(name)


def get_probe_class(name: str) -> type:
    # Side-effect imports
    from llm4rec.components.evaluation.probes import (  # noqa: F401
        framing,
        permutation,
        popularity,
        position,
        recency,
        semantic_prefix,
    )

    return PROBE_REGISTRY.get(name)


def build_probe(name: str, **kwargs: Any) -> BiasProbe:
    return get_probe_class(name)(**kwargs)


def build_probes_from_config(bias_cfg: dict[str, Any] | None) -> list[BiasProbe]:
    cfg = bias_cfg or {}
    names = list(cfg.get("probes") or ["popularity", "position", "framing", "recency"])
    probes: list[BiasProbe] = []
    for name in names:
        key = str(name).strip().lower()
        if key == "history_reversal":
            # Folded into recency via reverse_history intervention
            key = "recency"
        sub = dict(cfg.get(key) or {})
        probes.append(build_probe(key, **sub))
    # Deduplicate by name while preserving order
    seen: set[str] = set()
    unique: list[BiasProbe] = []
    for p in probes:
        if p.name in seen:
            continue
        seen.add(p.name)
        unique.append(p)
    return unique
