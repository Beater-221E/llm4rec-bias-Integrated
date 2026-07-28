"""MovieLens dataset builders (classic 100K / 1M for Letter & SID routes)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.components.dataset.base import DatasetBuilder, DatasetBundle
from llm4rec.components.dataset.processor import DatasetProcessor
from llm4rec.components.dataset.schema import InteractionSchema
from llm4rec.components.dataset._impl.movielens.ml100k import MovieLens100KAdapter
from llm4rec.components.dataset._impl.movielens.ml1m import MovieLens1MAdapter
from llm4rec.components.dataset._impl.registry import build_dataset
from llm4rec.core.paths import project_root
from llm4rec.core.schemas import RecommendationExample, TaskSpec


def _adapter_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config or {})
    data_root = cfg.pop("data_root", None)
    if data_root is None:
        data_root = project_root() / "data"
    else:
        data_root = Path(data_root)
        if not data_root.is_absolute():
            data_root = project_root() / data_root
    return {
        "data_root": data_root,
        "rating_threshold": float(cfg.get("rating_threshold", 4.0)),
        "split": str(cfg.get("split", "leave_one_out")),
        "history_max_length": int(cfg.get("history_max_length", 20)),
        "candidate_size": int(cfg.get("candidate_size", 10)),
        "negative_sampling": str(cfg.get("negative_sampling", "uniform")),
        "target_position": str(cfg.get("target_position", "random")),
        "framing": str(cfg.get("framing", "neutral")),
        "min_user_interactions": int(cfg.get("min_user_interactions", 5)),
        "seed": int(cfg.get("seed", 42)),
        "train_limit": cfg.get("train_limit"),
        "eval_limit": cfg.get("eval_limit"),
        "train_ratio": float(cfg.get("train_ratio", 0.8)),
        "val_ratio": float(cfg.get("val_ratio", 0.1)),
    }


class _AdapterBuilder(DatasetBuilder):
    """Wrap an existing MovieLens DatasetAdapter as a DatasetBuilder."""

    adapter_cls: type
    name: str
    schema = InteractionSchema()

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        cfg = dict(config or {})
        cfg.update(kwargs)
        self.config = cfg
        self._adapter = None

    def _make_adapter(self):
        if self._adapter is None:
            self._adapter = build_dataset(self.name, **_adapter_kwargs(self.config))
        return self._adapter

    def prepare(self) -> None:
        adapter = self._make_adapter()
        adapter.download()
        adapter.preprocess()

    def build(self) -> DatasetBundle:
        adapter = self._make_adapter()
        try:
            interactions = adapter.load_interactions()
        except Exception:
            self.prepare()
            interactions = adapter.load_interactions()

        users, items = DatasetProcessor.unique_ids(interactions)
        sequences = DatasetProcessor.build_sequences(
            interactions,
            max_len=int(self.config.get("history_max_length") or 0) or None,
        )
        meta: dict[str, Any] = {
            "schema": self.schema.to_dict(),
            "adapter": self.name,
            "fingerprint": adapter.fingerprint(),
        }
        summary = adapter.summary() if hasattr(adapter, "summary") else {}
        if isinstance(summary, dict):
            meta.update(summary)
        extras: dict[str, Any] = {}
        processed = getattr(adapter, "processed_dir", None)
        if processed is not None:
            extras["processed_dir"] = Path(processed)
        return DatasetBundle(
            name=self.name,
            interactions=list(interactions),
            users=users,
            items=items,
            sequences=sequences,
            metadata=meta,
            extras=extras,
        )

    def build_examples(self, split: str, task_spec: TaskSpec) -> list[RecommendationExample]:
        adapter = self._make_adapter()
        return adapter.build_examples(split, task_spec)


class MovieLens100KBuilder(_AdapterBuilder):
    """Classic GroupLens MovieLens-100K (``u.data`` / ``u.item``)."""

    name = "movielens_100k"
    adapter_cls = MovieLens100KAdapter


class MovieLens1MBuilder(_AdapterBuilder):
    """MovieLens-1M."""

    name = "movielens_1m"
    adapter_cls = MovieLens1MAdapter


_BUILDERS: dict[str, type[DatasetBuilder]] = {
    "movielens_100k": MovieLens100KBuilder,
    "ml100k": MovieLens100KBuilder,
    "movielens_1m": MovieLens1MBuilder,
    "ml1m": MovieLens1MBuilder,
}


def build_movielens_bundle(
    name: str,
    config: dict[str, Any] | None = None,
    *,
    prepare: bool = False,
) -> DatasetBundle:
    """Factory helper for MovieLens DatasetBundle construction."""
    key = name.strip().lower().replace("-", "_")
    # Map short aliases to registry names
    alias = {"ml100k": "movielens_100k", "ml1m": "movielens_1m"}
    registry_name = alias.get(key, key)
    cls = _BUILDERS.get(key) or _BUILDERS.get(registry_name)
    if cls is None:
        raise KeyError(f"Unknown MovieLens dataset '{name}'")
    builder = cls(config)
    builder.name = registry_name  # type: ignore[misc]
    if prepare:
        builder.prepare()
    return builder.build()


__all__ = [
    "MovieLens100KBuilder",
    "MovieLens1MBuilder",
    "build_movielens_bundle",
]
