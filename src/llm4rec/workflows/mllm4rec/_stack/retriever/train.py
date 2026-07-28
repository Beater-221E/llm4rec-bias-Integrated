# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
# (trainer/lru.py + trainer/base.py + train_retriever.py)
#
# Original behavior is preserved unless explicitly documented.

"""Train LRU retriever and export official-compatible retrieved.pkl."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from llm4rec.workflows.mllm4rec._stack.metrics import absolute_recall_mrr_ndcg_for_ks
from llm4rec.workflows.mllm4rec._stack.retriever.data import build_lru_loaders, load_official_dataset
from llm4rec.workflows.mllm4rec._stack.retriever.model import LRURec

logger = logging.getLogger("llm4rec.workflows.mllm4rec._stack.retriever")


@dataclass
class RetrieverConfig:
    dataset_pkl: Path
    export_root: Path
    device: str = "cuda"
    seed: int = 42
    # ml-100k official set_template defaults
    bert_max_len: int = 200
    bert_hidden_units: int = 64
    bert_num_blocks: int = 2
    bert_num_heads: int = 2
    bert_head_size: int = 32
    bert_dropout: float = 0.2
    bert_attn_dropout: float = 0.2
    sliding_window_size: float = 1.0
    train_batch_size: int = 16
    val_batch_size: int = 16
    test_batch_size: int = 16
    num_workers: int = 0
    num_epochs: int = 500
    lr: float = 0.001
    weight_decay: float = 0.01
    max_grad_norm: float = 5.0
    val_strategy: str = "iteration"  # iteration | epoch
    val_iterations: int = 500
    early_stopping: bool = True
    early_stopping_patience: int = 20
    metric_ks: tuple[int, ...] = (1, 5, 10, 20, 50)
    best_metric: str = "Recall@10"
    num_items: int = 0  # filled at runtime


def _seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch, device: str):
    return tuple(x.to(device) for x in batch)


def _eval_metrics(model, loader, device, metric_ks, exclude_history: bool) -> dict[str, float]:
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            seqs, labels = _to_device(batch, device)
            scores = model(seqs)[:, -1, :]
            if exclude_history:
                b, l = seqs.shape
                for i in range(l):
                    scores[torch.arange(scores.size(0), device=device), seqs[:, i]] = -1e9
                scores[:, 0] = -1e9
            all_scores.append(scores.cpu())
            all_labels.append(labels.view(-1).cpu())
    scores_t = torch.cat(all_scores, dim=0)
    labels_t = torch.cat(all_labels, dim=0)
    return absolute_recall_mrr_ndcg_for_ks(scores_t, labels_t, list(metric_ks))


def generate_candidates(
    model: nn.Module,
    val_loader,
    test_loader,
    *,
    device: str,
    metric_ks: list[int],
    retrieved_path: Path,
) -> dict[str, Any]:
    """Official LRUTrainer.generate_candidates → retrieved.pkl schema."""
    model.eval()
    val_probs, val_labels = [], []
    test_probs, test_labels = [], []
    with torch.no_grad():
        logger.info("Generating candidates for validation set")
        for batch in tqdm(val_loader, desc="retrieved/val"):
            seqs, labels = _to_device(batch, device)
            scores = model(seqs)[:, -1, :]
            b, l = seqs.shape
            for i in range(l):
                scores[torch.arange(scores.size(0), device=device), seqs[:, i]] = -1e9
            scores[:, 0] = -1e9
            val_probs.extend(scores.cpu().tolist())
            val_labels.extend(labels.view(-1).cpu().tolist())
        val_metrics = absolute_recall_mrr_ndcg_for_ks(
            torch.tensor(val_probs), torch.tensor(val_labels).view(-1), metric_ks
        )
        logger.info("val_metrics=%s", val_metrics)

        logger.info("Generating candidates for test set")
        for batch in tqdm(test_loader, desc="retrieved/test"):
            seqs, labels = _to_device(batch, device)
            scores = model(seqs)[:, -1, :]
            b, l = seqs.shape
            for i in range(l):
                scores[torch.arange(scores.size(0), device=device), seqs[:, i]] = -1e9
            scores[:, 0] = -1e9
            test_probs.extend(scores.cpu().tolist())
            test_labels.extend(labels.view(-1).cpu().tolist())
        test_metrics = absolute_recall_mrr_ndcg_for_ks(
            torch.tensor(test_probs), torch.tensor(test_labels).view(-1), metric_ks
        )
        logger.info("test_metrics=%s", test_metrics)

    payload = {
        "val_probs": val_probs,
        "val_labels": val_labels,
        "val_metrics": val_metrics,
        "test_probs": test_probs,
        "test_labels": test_labels,
        "test_metrics": test_metrics,
    }
    retrieved_path.parent.mkdir(parents=True, exist_ok=True)
    with retrieved_path.open("wb") as f:
        pickle.dump(payload, f)
    logger.info("Wrote %s", retrieved_path)
    return payload


def train_retriever(cfg: RetrieverConfig) -> Path:
    from llm4rec.tracking.inplace_progress import (
        install_inplace_progress,
        write_progress_status,
    )

    install_inplace_progress(force=True)
    write_progress_status("retriever: loading…")
    _seed_everything(cfg.seed)
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    dataset = load_official_dataset(cfg.dataset_pkl)
    train_loader, val_loader, test_loader, num_users, num_items = build_lru_loaders(
        dataset,
        max_len=cfg.bert_max_len,
        sliding_window_size=cfg.sliding_window_size,
        train_batch_size=cfg.train_batch_size,
        val_batch_size=cfg.val_batch_size,
        test_batch_size=cfg.test_batch_size,
        num_workers=cfg.num_workers,
    )
    cfg.num_items = num_items
    logger.info("users=%s items=%s train_batches=%s", num_users, num_items, len(train_loader))

    args = SimpleNamespace(**{k: getattr(cfg, k) for k in asdict(cfg)})
    args.num_items = num_items
    model = LRURec(args).to(cfg.device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss(ignore_index=0)

    export = Path(cfg.export_root)
    models_dir = export / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_path = models_dir / "best_acc_model.pth"
    best_metric_val = -1.0
    patience_left = cfg.early_stopping_patience
    accum_iter = 0
    stop = False

    # initial validate (official)
    metrics = _eval_metrics(
        model, val_loader, cfg.device, list(cfg.metric_ks), exclude_history=False
    )
    logger.info("initial val %s", {cfg.best_metric: metrics.get(cfg.best_metric)})

    for epoch in range(cfg.num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.num_epochs}")
        for batch in pbar:
            seqs, labels = _to_device(batch, cfg.device)
            optimizer.zero_grad()
            logits = model(seqs)
            loss = ce(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            accum_iter += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}")

            if cfg.val_strategy == "iteration" and accum_iter % cfg.val_iterations == 0:
                metrics = _eval_metrics(
                    model, val_loader, cfg.device, list(cfg.metric_ks), exclude_history=False
                )
                cur = metrics.get(cfg.best_metric, 0.0)
                logger.info("iter %s val %s=%.4f", accum_iter, cfg.best_metric, cur)
                if cur > best_metric_val:
                    best_metric_val = cur
                    patience_left = cfg.early_stopping_patience
                    torch.save({"model_state_dict": model.state_dict()}, best_path)
                elif cfg.early_stopping:
                    patience_left -= 1
                    if patience_left <= 0:
                        stop = True
                        break
        if stop:
            logger.info("Early stopping")
            break
        # Always validate at epoch end (official also supports epoch strategy;
        # this ensures short smoke runs still checkpoint).
        metrics = _eval_metrics(
            model, val_loader, cfg.device, list(cfg.metric_ks), exclude_history=False
        )
        cur = metrics.get(cfg.best_metric, 0.0)
        logger.info("epoch %s val %s=%.4f", epoch + 1, cfg.best_metric, cur)
        if cur > best_metric_val:
            best_metric_val = cur
            patience_left = cfg.early_stopping_patience
            torch.save({"model_state_dict": model.state_dict()}, best_path)
        elif cfg.early_stopping and cfg.val_strategy == "epoch":
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping")
                break

    if not best_path.is_file():
        torch.save({"model_state_dict": model.state_dict()}, best_path)
    state = torch.load(best_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    test_metrics = _eval_metrics(
        model, test_loader, cfg.device, list(cfg.metric_ks), exclude_history=True
    )
    (export / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    logger.info("test_metrics=%s", test_metrics)

    retrieved = export / "retrieved.pkl"
    generate_candidates(
        model,
        val_loader,
        test_loader,
        device=cfg.device,
        metric_ks=list(cfg.metric_ks),
        retrieved_path=retrieved,
    )
    return retrieved
