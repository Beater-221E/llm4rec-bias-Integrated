# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
# (dataloader/lru.py)
#
# Original behavior is preserved unless explicitly documented.

"""LRU retriever dataloaders over official-compatible dataset.pkl."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data as data_utils

from llm4rec.workflows.mllm4rec.data.serializer import load_pickle

logger = logging.getLogger("llm4rec.workflows.mllm4rec._stack.retriever")


def worker_init_fn(worker_id: int) -> None:
    random.seed(np.random.get_state()[1][0] + worker_id)
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def load_official_dataset(path: str | Path) -> dict[str, Any]:
    ds = load_pickle(path)
    for key in ("train", "val", "test", "umap", "smap"):
        if key not in ds:
            raise KeyError(f"dataset.pkl missing {key}")
    return ds


class LRUTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len: int, sliding_size: float, num_items: int):
        self.max_len = max_len
        self.sliding_step = int(sliding_size * max_len)
        self.num_items = num_items
        assert self.sliding_step > 0
        self.all_seqs: list[list[int]] = []
        for u in sorted(u2seq.keys()):
            seq = u2seq[u]
            if len(seq) < self.max_len + self.sliding_step:
                self.all_seqs.append(seq)
            else:
                start_idx = range(len(seq) - max_len, -1, -self.sliding_step)
                self.all_seqs = self.all_seqs + [seq[i : i + max_len] for i in start_idx]

    def __len__(self) -> int:
        return len(self.all_seqs)

    def __getitem__(self, index: int):
        seq = self.all_seqs[index]
        labels = seq[-self.max_len :]
        tokens = seq[:-1][-self.max_len :]
        tokens = [0] * (self.max_len - len(tokens)) + tokens
        labels = [0] * (self.max_len - len(labels)) + labels
        return torch.LongTensor(tokens), torch.LongTensor(labels)


class LRUValidDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len: int):
        self.u2seq = u2seq
        self.u2answer = u2answer
        users = sorted(self.u2seq.keys())
        self.users = [u for u in users if len(u2answer[u]) > 0]
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int):
        user = self.users[index]
        seq = self.u2seq[user][-self.max_len :]
        seq = [0] * (self.max_len - len(seq)) + seq
        return torch.LongTensor(seq), torch.LongTensor(self.u2answer[user])


class LRUTestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2val, u2answer, max_len: int, subset_users=None):
        self.u2seq = u2seq
        self.u2val = u2val
        self.u2answer = u2answer
        users = sorted(self.u2seq.keys())
        self.users = [
            u for u in users if len(u2val[u]) > 0 and len(u2answer[u]) > 0
        ]
        self.max_len = max_len
        if subset_users is not None:
            self.users = subset_users

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int):
        user = self.users[index]
        seq = (self.u2seq[user] + self.u2val[user])[-self.max_len :]
        seq = [0] * (self.max_len - len(seq)) + seq
        return torch.LongTensor(seq), torch.LongTensor(self.u2answer[user])


def build_lru_loaders(
    dataset: dict[str, Any],
    *,
    max_len: int,
    sliding_window_size: float,
    train_batch_size: int,
    val_batch_size: int,
    test_batch_size: int,
    num_workers: int = 0,
):
    train, val, test = dataset["train"], dataset["val"], dataset["test"]
    num_users = len(dataset["umap"])
    num_items = len(dataset["smap"])
    train_ds = LRUTrainDataset(train, max_len, sliding_window_size, num_items)
    val_ds = LRUValidDataset(train, val, max_len)
    test_ds = LRUTestDataset(train, val, test, max_len)
    train_loader = data_utils.DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )
    val_loader = data_utils.DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = data_utils.DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, num_users, num_items
