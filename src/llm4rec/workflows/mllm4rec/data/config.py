# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
#
# Original behavior is preserved unless explicitly documented.

"""YAML / CLI / env configuration for the MLLM4Rec data pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from llm4rec.workflows.mllm4rec.data.constants import (
    DEFAULT_BLIP2_MODEL,
    DEFAULT_MIN_RATING,
    DEFAULT_MIN_SC,
    DEFAULT_MIN_UC,
    DEFAULT_SEED,
    OFFICIAL_ML100K_CODE,
    TMDB_API_KEY_ENV,
    TMDB_IMAGE_BASE_URL,
    preprocessed_folder_name,
)

CompatibilityMode = Literal["original", "robust"]


@dataclass
class MLLM4RecDataConfig:
    """Resolved config for staged data generation."""

    dataset_code: str = OFFICIAL_ML100K_CODE
    raw_dir: Path = Path("data/raw/ml-100k")
    output_root: Path = Path("data/preprocessed")
    min_rating: int = DEFAULT_MIN_RATING
    min_uc: int = DEFAULT_MIN_UC
    min_sc: int = DEFAULT_MIN_SC
    seed: int = DEFAULT_SEED

    compatibility_mode: CompatibilityMode = "original"
    filtering_mode: CompatibilityMode = "original"
    iterative_kcore: bool = True

    tmdb_api_key_env: str = TMDB_API_KEY_ENV
    tmdb_match_mode: CompatibilityMode = "original"
    tmdb_image_base_url: str = TMDB_IMAGE_BASE_URL
    tmdb_timeout_seconds: int = 30
    tmdb_retries: int = 3
    tmdb_max_workers: int = 8
    tmdb_resume: bool = True

    caption_model_name_or_path: str = DEFAULT_BLIP2_MODEL
    caption_device: str = "cuda"
    caption_dtype: str = "float16"
    caption_mode: str = "original"
    caption_batch_size: int = 1
    caption_resume: bool = True

    save_pickle: bool = True
    save_parquet: bool = True
    atomic_write: bool = True
    create_backup: bool = True

    log_level: str = "INFO"
    log_dir: Path = Path("logs/mllm4rec")

    overwrite: bool = False
    resume: bool = True
    max_items: int | None = None
    retry_failed_only: bool = False

    @property
    def preprocessed_dir(self) -> Path:
        name = preprocessed_folder_name(
            code=self.dataset_code,
            min_rating=self.min_rating,
            min_uc=self.min_uc,
            min_sc=self.min_sc,
        )
        return Path(self.output_root) / name

    @property
    def dataset_pkl_path(self) -> Path:
        return self.preprocessed_dir / "dataset.pkl"

    @property
    def img_dir(self) -> Path:
        return self.preprocessed_dir / "img"


def _as_path(value: Any, default: Path) -> Path:
    if value is None:
        return Path(default)
    return Path(value)


def load_data_config(
    path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> MLLM4RecDataConfig:
    """Load YAML config and apply flat overrides (CLI)."""
    raw: dict[str, Any] = {}
    if path is not None:
        with Path(path).open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    ds = raw.get("dataset") or {}
    compat = raw.get("compatibility") or {}
    filtering = raw.get("filtering") or {}
    tmdb = raw.get("tmdb") or {}
    caption = raw.get("caption") or {}
    serialization = raw.get("serialization") or {}
    logging_cfg = raw.get("logging") or {}

    cfg = MLLM4RecDataConfig(
        dataset_code=str(ds.get("code", OFFICIAL_ML100K_CODE)),
        raw_dir=_as_path(ds.get("raw_dir"), Path("data/raw/ml-100k")),
        output_root=_as_path(ds.get("output_root"), Path("data/preprocessed")),
        min_rating=int(ds.get("min_rating", DEFAULT_MIN_RATING)),
        min_uc=int(ds.get("min_uc", DEFAULT_MIN_UC)),
        min_sc=int(ds.get("min_sc", DEFAULT_MIN_SC)),
        seed=int(ds.get("seed", DEFAULT_SEED)),
        compatibility_mode=str(compat.get("mode", "original")),  # type: ignore[arg-type]
        filtering_mode=str(filtering.get("mode", compat.get("mode", "original"))),  # type: ignore[arg-type]
        iterative_kcore=bool(filtering.get("iterative_kcore", True)),
        tmdb_api_key_env=str(tmdb.get("api_key_env", TMDB_API_KEY_ENV)),
        tmdb_match_mode=str(tmdb.get("match_mode", "original")),  # type: ignore[arg-type]
        tmdb_image_base_url=str(tmdb.get("image_base_url", TMDB_IMAGE_BASE_URL)),
        tmdb_timeout_seconds=int(tmdb.get("timeout_seconds", 30)),
        tmdb_retries=int(tmdb.get("retries", 3)),
        tmdb_max_workers=int(tmdb.get("max_workers", 8)),
        tmdb_resume=bool(tmdb.get("resume", True)),
        caption_model_name_or_path=str(
            caption.get("model_name_or_path", DEFAULT_BLIP2_MODEL)
        ),
        caption_device=str(caption.get("device", "cuda")),
        caption_dtype=str(caption.get("dtype", "float16")),
        caption_mode=str(caption.get("mode", "original")),
        caption_batch_size=int(caption.get("batch_size", 1)),
        caption_resume=bool(caption.get("resume", True)),
        save_pickle=bool(serialization.get("save_pickle", True)),
        save_parquet=bool(serialization.get("save_parquet", True)),
        atomic_write=bool(serialization.get("atomic_write", True)),
        create_backup=bool(serialization.get("create_backup", True)),
        log_level=str(logging_cfg.get("level", "INFO")),
        log_dir=_as_path(logging_cfg.get("log_dir"), Path("logs/mllm4rec")),
    )

    if overrides:
        for key, value in overrides.items():
            if value is None or not hasattr(cfg, key):
                continue
            if key in {"raw_dir", "output_root", "log_dir"}:
                value = Path(value)
            cfg = replace(cfg, **{key: value})
    return cfg
