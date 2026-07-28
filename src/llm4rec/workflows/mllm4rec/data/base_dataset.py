# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
#
# Original behavior is preserved unless explicitly documented.

"""Abstract dataset base (official AbstractDataset port)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from llm4rec.workflows.mllm4rec.data.config import MLLM4RecDataConfig
from llm4rec.workflows.mllm4rec.data.filtering import filter_triplets
from llm4rec.workflows.mllm4rec.data.id_mapping import assert_maps_invertible, densify_index
from llm4rec.workflows.mllm4rec.data.serializer import (
    load_pickle,
    save_id_map_sidecars,
    save_pickle,
    save_statistics_json,
    try_save_parquet_sidecars,
)
from llm4rec.workflows.mllm4rec.data.splitting import split_df

logger = logging.getLogger("llm4rec.workflows.mllm4rec._stack")


class BaseMovieLensDataset(ABC):
    """Shared MovieLens preprocess pipeline for MLLM4Rec-compatible pickles."""

    def __init__(self, cfg: MLLM4RecDataConfig) -> None:
        self.cfg = cfg
        self.min_rating = cfg.min_rating
        self.min_uc = cfg.min_uc
        self.min_sc = cfg.min_sc
        if self.min_uc < 2:
            raise AssertionError(
                "Need at least 2 ratings per user for validation and test"
            )

    @classmethod
    @abstractmethod
    def code(cls) -> str:
        raise NotImplementedError

    @classmethod
    def raw_code(cls) -> str:
        return cls.code()

    @classmethod
    @abstractmethod
    def url(cls) -> str:
        raise NotImplementedError

    @classmethod
    def zip_file_content_is_folder(cls) -> bool:
        return True

    @classmethod
    @abstractmethod
    def all_raw_file_names(cls) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def maybe_download_raw_dataset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_ratings_df(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def load_meta_dict(self) -> dict[Any, str]:
        raise NotImplementedError

    def raw_folder_path(self) -> Path:
        return Path(self.cfg.raw_dir)

    def preprocessed_folder_path(self) -> Path:
        return self.cfg.preprocessed_dir

    def preprocessed_dataset_path(self) -> Path:
        return self.cfg.dataset_pkl_path

    def load_dataset(self) -> dict[str, Any]:
        self.preprocess()
        return load_pickle(self.preprocessed_dataset_path())

    def preprocess(self, *, overwrite: bool | None = None) -> Path:
        """Build official-compatible ``dataset.pkl`` (no captions yet)."""
        overwrite = self.cfg.overwrite if overwrite is None else overwrite
        dataset_path = self.preprocessed_dataset_path()
        if dataset_path.is_file() and not overwrite:
            logger.info("Already preprocessed. Skip preprocessing (%s)", dataset_path)
            return dataset_path

        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        if self.cfg.compatibility_mode == "robust" and "source_row_index" not in df.columns:
            df = df.copy()
            df["source_row_index"] = range(len(df))

        meta_raw = self.load_meta_dict()
        df = df[df["sid"].isin(meta_raw)]
        df = filter_triplets(
            df,
            min_uc=self.min_uc,
            min_sc=self.min_sc,
            min_rating=self.min_rating,
            mode=self.cfg.filtering_mode,  # type: ignore[arg-type]
            iterative_kcore=self.cfg.iterative_kcore,
        )
        df, umap, smap = densify_index(df)
        assert_maps_invertible(umap, smap)
        train, val, test = split_df(
            df,
            len(umap),
            mode=self.cfg.compatibility_mode,  # type: ignore[arg-type]
        )
        meta = {smap[k]: v for k, v in meta_raw.items() if k in smap}
        dataset: dict[str, Any] = {
            "train": train,
            "val": val,
            "test": test,
            "meta": meta,
            "umap": umap,
            "smap": smap,
        }

        if self.cfg.save_pickle:
            save_pickle(
                dataset,
                dataset_path,
                atomic_write=self.cfg.atomic_write,
                create_backup=self.cfg.create_backup,
            )
        save_id_map_sidecars(dataset_path.parent, umap=umap, smap=smap)
        if self.cfg.save_parquet:
            try_save_parquet_sidecars(dataset_path.parent, dataset)
        save_statistics_json(
            dataset_path.parent,
            compute_preprocess_statistics(dataset),
        )
        return dataset_path


def compute_preprocess_statistics(dataset: dict[str, Any]) -> dict[str, Any]:
    train = dataset["train"]
    lengths = [len(train[u]) + len(dataset["val"][u]) + len(dataset["test"][u]) for u in train]
    n_users = len(dataset["umap"])
    n_items = len(dataset["smap"])
    n_interactions = sum(lengths)
    return {
        "num_users": n_users,
        "num_items": n_items,
        "num_interactions": n_interactions,
        "average_sequence_length": (sum(lengths) / n_users) if n_users else 0.0,
        "minimum_sequence_length": min(lengths) if lengths else 0,
        "maximum_sequence_length": max(lengths) if lengths else 0,
        "sparsity": 1.0 - (n_interactions / (n_users * n_items)) if n_users and n_items else None,
        "has_meta_img_des": "meta_img_des" in dataset,
        "meta_size": len(dataset.get("meta") or {}),
    }
