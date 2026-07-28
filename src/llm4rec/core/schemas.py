"""Shared data contracts for datasets, training, evaluation, and probes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Interaction:
    """One user–item event after normalization."""

    user_id: str
    item_id: str
    rating: float | None
    timestamp: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecommendationExample:
    """Standardized recommendation sample consumed by workflows."""

    example_id: str
    user_id: str
    history_item_ids: list[str]
    target_item_id: str
    candidates: list[str] | None
    prompt_messages: list[dict[str, str]]
    target_text: str
    target_index: int | None = None
    semantic_id: list[int] | None = None
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSplits:
    """Train / validation / test interaction partitions."""

    train: list[Interaction]
    validation: list[Interaction]
    test: list[Interaction]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSpec:
    """How examples should be constructed for a workflow stage."""

    task: str
    history_max_length: int = 20
    candidate_size: int = 10
    negative_sampling: str = "uniform"
    target_position: str = "random"
    framing: str = "neutral"
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardOutput:
    """Composable reward with per-component tensors and scalar telemetry."""

    total: Any  # torch.Tensor when training stack is installed
    components: dict[str, Any]
    telemetry: dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Unified evaluation payload.

    Metric values may be ``float`` or the sentinel ``\"not_applicable\"``
    (never silently zero for unsupported metrics).
    """

    metrics: dict[str, float | str]
    slices: dict[str, dict[str, float | str | int]]
    predictions_path: Path | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbePair:
    """Original vs counterfactual example for a bias probe."""

    original: RecommendationExample
    counterfactual: RecommendationExample
    manipulated_signal: str
    held_constant: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeResult:
    """Probe metrics and optional artifacts."""

    name: str
    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)
