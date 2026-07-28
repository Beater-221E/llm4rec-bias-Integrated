"""Bias / shortcut probes (Phase 6)."""

from __future__ import annotations

from llm4rec.components.evaluation.probes.base import BiasProbe
from llm4rec.components.evaluation.probes.registry import build_probe, build_probes_from_config, register_probe

__all__ = [
    "BiasProbe",
    "build_probe",
    "build_probes_from_config",
    "register_probe",
]
