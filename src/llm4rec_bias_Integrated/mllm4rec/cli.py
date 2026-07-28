"""CLI for MLLM4Rec Retriever + Ranker training.

Usage::

    python -m llm4rec_bias_Integrated.mllm4rec.cli train-retriever \\
      --config configs/training/mllm4rec_retriever.yaml

    python -m llm4rec_bias_Integrated.mllm4rec.cli train-ranker \\
      --config configs/training/mllm4rec_ranker.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import yaml

logger = logging.getLogger("llm4rec_bias_Integrated.mllm4rec")


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
    )


def _load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def cmd_train_retriever(args: argparse.Namespace) -> int:
    from llm4rec_bias_Integrated.mllm4rec.retriever.train import RetrieverConfig, train_retriever

    _setup_logging(args.log_level)
    raw = _load_yaml(args.config)
    ret = raw.get("retriever") or {}
    ds = raw.get("dataset") or {}

    dataset_pkl = Path(
        args.dataset_pkl
        or ret.get("dataset_pkl")
        or ds.get("dataset_pkl")
        or "data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/dataset.pkl"
    )
    export_root = Path(
        args.export_root
        or ret.get("export_root")
        or "experiments/lru/ml-100k"
    )
    cfg = RetrieverConfig(
        dataset_pkl=dataset_pkl,
        export_root=export_root,
        device=args.device or ret.get("device", "cuda"),
        seed=int(ret.get("seed", 42)),
        bert_max_len=int(ret.get("bert_max_len", 200)),
        train_batch_size=int(ret.get("train_batch_size", 16)),
        val_batch_size=int(ret.get("val_batch_size", 16)),
        test_batch_size=int(ret.get("test_batch_size", 16)),
        num_epochs=int(args.num_epochs or ret.get("num_epochs", 500)),
        val_iterations=int(ret.get("val_iterations", 500)),
        val_strategy=str(ret.get("val_strategy", "iteration")),
        early_stopping_patience=int(ret.get("early_stopping_patience", 20)),
        lr=float(ret.get("lr", 0.001)),
        weight_decay=float(ret.get("weight_decay", 0.01)),
        bert_dropout=float(ret.get("bert_dropout", 0.2)),
        bert_attn_dropout=float(ret.get("bert_attn_dropout", 0.2)),
    )
    path = train_retriever(cfg)
    logger.info("Retriever done. retrieved.pkl → %s", path)
    return 0


def cmd_train_ranker(args: argparse.Namespace) -> int:
    from llm4rec_bias_Integrated.mllm4rec.ranker.train import RankerConfig, train_ranker

    _setup_logging(args.log_level)
    raw = _load_yaml(args.config)
    rk = raw.get("ranker") or {}
    dataset_pkl = Path(
        args.dataset_pkl
        or rk.get("dataset_pkl")
        or "data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/dataset.pkl"
    )
    retrieved_pkl = Path(
        args.retrieved_pkl
        or rk.get("retrieved_pkl")
        or "experiments/lru/ml-100k/retrieved.pkl"
    )
    export_root = Path(args.export_root or rk.get("export_root") or "experiments/ranker/ml-100k")
    model = args.llm_base_model or rk.get("llm_base_model")
    if not model:
        logger.error("ranker.llm_base_model is required (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
        return 1
    cfg = RankerConfig(
        dataset_pkl=dataset_pkl,
        retrieved_pkl=retrieved_pkl,
        export_root=export_root,
        llm_base_model=model,
        llm_base_tokenizer=rk.get("llm_base_tokenizer") or model,
        device=args.device or rk.get("device", "cuda"),
        dtype=str(rk.get("dtype", "float16")),
        load_in_4bit=bool(rk.get("load_in_4bit", False)),
        lora_num_epochs=int(args.num_epochs or rk.get("lora_num_epochs", 3)),
        lora_micro_batch_size=int(rk.get("lora_micro_batch_size", 2)),
        lora_lr=float(rk.get("lora_lr", 1e-4)),
        max_train_steps=args.max_train_steps or rk.get("max_train_steps"),
    )
    if not retrieved_pkl.is_file():
        logger.error("Missing %s — run train-retriever first", retrieved_pkl)
        return 1
    out = train_ranker(cfg)
    logger.info("Ranker done → %s", out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m llm4rec_bias_Integrated.mllm4rec.cli")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=str, default=None)
    common.add_argument("--log-level", type=str, default="INFO")
    common.add_argument("--device", type=str, default=None)
    common.add_argument("--dataset-pkl", type=str, default=None)
    common.add_argument("--export-root", type=str, default=None)
    common.add_argument("--num-epochs", type=int, default=None)

    sub = p.add_subparsers(dest="command", required=True)
    pr = sub.add_parser("train-retriever", parents=[common])
    pr.set_defaults(func=cmd_train_retriever)

    pk = sub.add_parser("train-ranker", parents=[common])
    pk.add_argument("--retrieved-pkl", type=str, default=None)
    pk.add_argument("--llm-base-model", type=str, default=None)
    pk.add_argument("--max-train-steps", type=int, default=None)
    pk.set_defaults(func=cmd_train_ranker)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    from llm4rec_bias_Integrated.tracking.inplace_progress import install_inplace_progress

    install_inplace_progress()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
