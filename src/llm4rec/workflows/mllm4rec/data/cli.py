# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
#
# Original behavior is preserved unless explicitly documented.

"""CLI for staged MLLM4Rec data generation.

Usage::

    python -m llm4rec.workflows.mllm4rec.data.cli preprocess \\
      --config configs/dataset/mllm4rec_ml100k.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from llm4rec.workflows.mllm4rec.data.blip2_captioner import generate_captions_for_dataset
from llm4rec.workflows.mllm4rec.data.config import MLLM4RecDataConfig, load_data_config
from llm4rec.workflows.mllm4rec.data.constants import OFFICIAL_ML100K_CODE
from llm4rec.workflows.mllm4rec.data.dataset_factory import dataset_factory
from llm4rec.workflows.mllm4rec.data.poster_downloader import download_posters_from_matches
from llm4rec.workflows.mllm4rec.data.serializer import (
    load_pickle,
    save_pickle,
    try_save_parquet_sidecars,
)
from llm4rec.workflows.mllm4rec.data.tmdb_client import (
    TMDbAPIError,
    TMDbClient,
    load_match_cache,
    match_dataset_meta,
)
from llm4rec.workflows.mllm4rec.data.validator import validate_dataset_pkl

logger = logging.getLogger("llm4rec.workflows.mllm4rec._stack")


def _configure_logging(level: str, log_dir: Path | None = None) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8"))
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def _config_from_args(args: argparse.Namespace) -> MLLM4RecDataConfig:
    overrides: dict = {
        "overwrite": bool(args.overwrite),
        "resume": bool(args.resume),
        "log_level": args.log_level,
    }
    if getattr(args, "max_items", None) is not None:
        overrides["max_items"] = args.max_items
    if getattr(args, "retry_failed_only", False):
        overrides["retry_failed_only"] = True
    if getattr(args, "dataset_code", None):
        overrides["dataset_code"] = args.dataset_code
    if getattr(args, "raw_dir", None):
        overrides["raw_dir"] = args.raw_dir
    if getattr(args, "output_root", None):
        overrides["output_root"] = args.output_root

    cfg_path = getattr(args, "config", None)
    return load_data_config(cfg_path, overrides=overrides)


def _ensure_dataset(cfg: MLLM4RecDataConfig) -> dict:
    if not cfg.dataset_pkl_path.is_file():
        raise FileNotFoundError(
            f"Missing {cfg.dataset_pkl_path}; run preprocess first."
        )
    return load_pickle(cfg.dataset_pkl_path)


def cmd_download(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    _configure_logging(cfg.log_level, cfg.log_dir)
    if getattr(args, "dataset_code", None):
        cfg.dataset_code = args.dataset_code
    ds = dataset_factory(cfg)
    ds.maybe_download_raw_dataset()
    logger.info("Raw data ready under %s", cfg.raw_dir)
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    _configure_logging(cfg.log_level, cfg.log_dir)
    ds = dataset_factory(cfg)
    path = ds.preprocess(overwrite=cfg.overwrite)
    logger.info("Preprocess complete: %s", path)
    return 0


def cmd_match_tmdb(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    _configure_logging(cfg.log_level, cfg.log_dir)
    try:
        dataset = _ensure_dataset(cfg)
        client = TMDbClient(
            api_key_env=cfg.tmdb_api_key_env,
            image_base_url=cfg.tmdb_image_base_url,
            timeout_seconds=cfg.tmdb_timeout_seconds,
            retries=cfg.tmdb_retries,
            match_mode=cfg.tmdb_match_mode,  # type: ignore[arg-type]
        )
        cache_path = cfg.preprocessed_dir / "tmdb_matches.jsonl"
        match_dataset_meta(
            dataset,
            client=client,
            cache_path=cache_path,
            resume=cfg.resume and cfg.tmdb_resume,
            overwrite=cfg.overwrite,
            max_items=cfg.max_items,
        )
        return 0
    except (TMDbAPIError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1


def cmd_download_posters(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    _configure_logging(cfg.log_level, cfg.log_dir)
    try:
        _ensure_dataset(cfg)
        cache_path = cfg.preprocessed_dir / "tmdb_matches.jsonl"
        matches = load_match_cache(cache_path)
        if not matches:
            logger.error("No TMDb cache at %s — run match-tmdb first", cache_path)
            return 1
        mode = "original" if cfg.compatibility_mode == "original" else "robust"
        download_posters_from_matches(
            matches=matches,
            img_dir=cfg.img_dir,
            mode=mode,  # type: ignore[arg-type]
            timeout_seconds=cfg.tmdb_timeout_seconds,
            retries=cfg.tmdb_retries,
            overwrite=cfg.overwrite,
            max_items=cfg.max_items,
            failed_log=cfg.preprocessed_dir / "failed_posters.jsonl",
        )
        return 0
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1


def cmd_generate_captions(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    _configure_logging(cfg.log_level, cfg.log_dir)
    try:
        dataset = _ensure_dataset(cfg)
        generate_captions_for_dataset(
            dataset,
            img_dir=cfg.img_dir,
            model_name_or_path=cfg.caption_model_name_or_path,
            device=cfg.caption_device,
            dtype=cfg.caption_dtype,
            mode="original" if cfg.caption_mode == "original" else "batched",  # type: ignore[arg-type]
            batch_size=cfg.caption_batch_size,
            resume=cfg.resume and cfg.caption_resume,
            overwrite=cfg.overwrite,
            max_items=cfg.max_items,
            captions_path=cfg.preprocessed_dir / "captions.jsonl",
            start_index=int(getattr(args, "start_index", 0) or 0),
        )
        save_pickle(
            dataset,
            cfg.dataset_pkl_path,
            atomic_write=cfg.atomic_write,
            create_backup=cfg.create_backup,
        )
        logger.info(
            "Wrote meta_img_des (%s keys) -> %s",
            len(dataset.get("meta_img_des") or {}),
            cfg.dataset_pkl_path,
        )
        return 0
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1


def cmd_serialize(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    _configure_logging(cfg.log_level, cfg.log_dir)
    path = cfg.dataset_pkl_path
    if not path.is_file():
        logger.error("Missing %s — run preprocess first", path)
        return 1
    dataset = load_pickle(path)
    save_pickle(
        dataset,
        path,
        atomic_write=cfg.atomic_write,
        create_backup=cfg.create_backup,
    )
    if cfg.save_parquet:
        try_save_parquet_sidecars(path.parent, dataset)
    logger.info("Re-serialized %s", path)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    _configure_logging(cfg.log_level, cfg.log_dir)
    if not cfg.dataset_pkl_path.is_file():
        logger.error("Missing %s", cfg.dataset_pkl_path)
        return 1
    report = validate_dataset_pkl(
        cfg.dataset_pkl_path,
        require_captions=bool(getattr(args, "require_captions", False)),
    )
    logger.info("Validation ok=%s errors=%s", report["ok"], report["errors"])
    return 0 if report["ok"] else 1


def cmd_build(args: argparse.Namespace) -> int:
    """Full multimodal data build (preprocess → TMDb → posters → captions → validate)."""
    skip_mm = bool(getattr(args, "skip_multimodal", False))
    if cmd_download(args) != 0:
        return 1
    if cmd_preprocess(args) != 0:
        return 1
    if not skip_mm:
        if cmd_match_tmdb(args) != 0:
            return 1
        if cmd_download_posters(args) != 0:
            return 1
        if cmd_generate_captions(args) != 0:
            return 1
    if cmd_validate(args) != 0:
        return 1
    logger.info("Build finished (skip_multimodal=%s)", skip_mm)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m llm4rec.workflows.mllm4rec.data.cli",
        description="MLLM4Rec official-compatible data generation",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=str, default=None, help="YAML config path")
    common.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    common.add_argument("--overwrite", action="store_true", default=False)
    common.add_argument("--retry-failed-only", action="store_true", default=False)
    common.add_argument("--max-items", type=int, default=None)
    common.add_argument("--log-level", type=str, default="INFO")
    common.add_argument("--output-root", type=str, default=None)
    common.add_argument("--raw-dir", type=str, default=None)

    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", parents=[common], help="Download raw MovieLens")
    p_dl.add_argument("--dataset-code", type=str, default=OFFICIAL_ML100K_CODE)
    p_dl.set_defaults(func=cmd_download)

    p_pre = sub.add_parser("preprocess", parents=[common], help="Build dataset.pkl")
    p_pre.add_argument("--dataset-code", type=str, default=None)
    p_pre.set_defaults(func=cmd_preprocess)

    p_match = sub.add_parser("match-tmdb", parents=[common], help="TMDb match")
    p_match.set_defaults(func=cmd_match_tmdb)

    p_post = sub.add_parser("download-posters", parents=[common], help="Download posters")
    p_post.set_defaults(func=cmd_download_posters)

    p_cap = sub.add_parser("generate-captions", parents=[common], help="BLIP2 captions")
    p_cap.add_argument("--start-index", type=int, default=0)
    p_cap.set_defaults(func=cmd_generate_captions)

    p_ser = sub.add_parser("serialize", parents=[common], help="Re-write sidecars/pickle")
    p_ser.set_defaults(func=cmd_serialize)

    p_val = sub.add_parser("validate", parents=[common], help="Validate dataset.pkl")
    p_val.add_argument("--require-captions", action="store_true", default=False)
    p_val.set_defaults(func=cmd_validate)

    p_build = sub.add_parser(
        "build",
        parents=[common],
        help="Full pipeline (use --skip-multimodal for preprocess-only)",
    )
    p_build.add_argument("--dataset-code", type=str, default=None)
    p_build.add_argument(
        "--skip-multimodal",
        action="store_true",
        default=False,
        help="Only download+preprocess+validate (no TMDb/BLIP2)",
    )
    p_build.set_defaults(func=cmd_build)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from llm4rec.tracking.inplace_progress import install_inplace_progress

    install_inplace_progress()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
