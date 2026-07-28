"""MLLM4Rec retriever facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from llm4rec.core.paths import project_root
from llm4rec.workflows.mllm4rec._stack.retriever.train import (
    RetrieverConfig,
    train_retriever,
)


def _as_retriever_config(raw: dict[str, Any], dataset_pkl: Path | None) -> RetrieverConfig:
    nested = dict(raw.get("retriever") or {})
    ds = raw.get("dataset") or {}
    if "dataset_pkl" in ds:
        nested.setdefault("dataset_pkl", ds["dataset_pkl"])
    if dataset_pkl is not None:
        nested["dataset_pkl"] = str(dataset_pkl)
    # Resolve relative paths
    for key in ("dataset_pkl", "export_root"):
        if key in nested and nested[key] is not None:
            p = Path(nested[key])
            if not p.is_absolute():
                nested[key] = str(project_root() / p)
    fields = set(RetrieverConfig.__dataclass_fields__)
    kwargs = {k: v for k, v in nested.items() if k in fields}
    kwargs["dataset_pkl"] = Path(kwargs["dataset_pkl"])
    kwargs["export_root"] = Path(kwargs["export_root"])
    return RetrieverConfig(**kwargs)


def train_retriever_from_config(
    config_path: str | Path,
    *,
    dataset_pkl: Path | None = None,
) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root() / path
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(raw, dict)
    cfg = _as_retriever_config(raw, dataset_pkl)
    retrieved = train_retriever(cfg)
    metrics_path = Path(cfg.export_root) / "test_metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_path.is_file():
        import json

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "retrieved_pkl": str(retrieved),
        "export_root": str(cfg.export_root),
        "metrics": metrics,
    }


__all__ = ["train_retriever_from_config", "train_retriever", "RetrieverConfig"]
