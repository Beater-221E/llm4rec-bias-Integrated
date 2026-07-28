"""MLLM4Rec ranker facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from llm4rec.core.paths import project_root
from llm4rec.workflows.mllm4rec._stack.ranker.train import RankerConfig, train_ranker


def _as_ranker_config(
    raw: dict[str, Any],
    *,
    retrieved_pkl: Path | None,
    dataset_pkl: Path | None,
) -> RankerConfig:
    nested = dict(raw.get("ranker") or raw)
    if retrieved_pkl is not None:
        nested["retrieved_pkl"] = str(retrieved_pkl)
    if dataset_pkl is not None:
        nested["dataset_pkl"] = str(dataset_pkl)
    for key in ("dataset_pkl", "retrieved_pkl", "export_root"):
        if key in nested and nested[key] is not None:
            p = Path(nested[key])
            if not p.is_absolute():
                nested[key] = str(project_root() / p)
    fields = set(RankerConfig.__dataclass_fields__)
    kwargs = {k: v for k, v in nested.items() if k in fields}
    kwargs["dataset_pkl"] = Path(kwargs["dataset_pkl"])
    kwargs["retrieved_pkl"] = Path(kwargs["retrieved_pkl"])
    kwargs["export_root"] = Path(kwargs["export_root"])
    if "lora_target_modules" in kwargs and isinstance(kwargs["lora_target_modules"], list):
        kwargs["lora_target_modules"] = tuple(kwargs["lora_target_modules"])
    return RankerConfig(**kwargs)


def train_ranker_from_config(
    config_path: str | Path,
    *,
    retrieved_pkl: Path | None = None,
    dataset_pkl: Path | None = None,
) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root() / path
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(raw, dict)
    cfg = _as_ranker_config(raw, retrieved_pkl=retrieved_pkl, dataset_pkl=dataset_pkl)
    out = train_ranker(cfg)
    metrics_path = Path(cfg.export_root) / "test_metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_path.is_file():
        import json

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "export_root": str(out if isinstance(out, Path) else cfg.export_root),
        "metrics": metrics,
    }


__all__ = ["train_ranker_from_config", "train_ranker", "RankerConfig"]
