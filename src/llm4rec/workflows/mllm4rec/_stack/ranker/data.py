# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
# (dataloader/llm.py)
#
# Original behavior is preserved unless explicitly documented.

"""LLM Ranker datasets using retrieved.pkl + meta + meta_img_des."""

from __future__ import annotations

import logging
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data as data_utils
from transformers import AutoTokenizer

from llm4rec.workflows.mllm4rec.data.serializer import load_pickle
from llm4rec.workflows.mllm4rec._stack.metrics import absolute_recall_mrr_ndcg_for_ks
from llm4rec.workflows.mllm4rec._stack.ranker.prompts import (
    DEFAULT_INPUT_TEMPLATE,
    DEFAULT_ML100K_SYSTEM,
    Prompter,
    format_seq_and_candidates,
)

logger = logging.getLogger("llm4rec.workflows.mllm4rec._stack.ranker")


def worker_init_fn(worker_id: int) -> None:
    random.seed(np.random.get_state()[1][0] + worker_id)
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def _tokenize_train(tokenizer, prompt: str, max_text_len: int, train_on_inputs: bool):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_text_len,
        padding=False,
        return_tensors=None,
    )
    if result["input_ids"][-1] != tokenizer.eos_token_id:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    if not train_on_inputs:
        # official: mask all but last 2 tokens (answer letter + eos region)
        result["labels"][:-2] = [-100] * len(result["labels"][:-2])
    return result


def _tokenize_eval(tokenizer, prompt: str, max_text_len: int, output_letter: str):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_text_len,
        padding=False,
        return_tensors=None,
    )
    result["labels"] = ord(output_letter) - ord("A")
    return result


def seq_to_token_ids(
    args,
    seq,
    candidates,
    label,
    text_dict,
    text_img_dict,
    tokenizer,
    prompter: Prompter,
    *,
    eval_mode: bool,
):
    dp = format_seq_and_candidates(
        seq=seq,
        candidates=candidates,
        label=label,
        text_dict=text_dict,
        text_img_dict=text_img_dict,
        tokenizer=tokenizer,
        max_title_len=args.llm_max_title_len,
        system_template=args.llm_system_template,
        input_template=args.llm_input_template,
    )
    if eval_mode:
        in_prompt = prompter.generate_prompt(dp["system"], dp["input"])
        return _tokenize_eval(
            tokenizer, in_prompt, args.llm_max_text_len, dp["output"]
        )
    full_prompt = prompter.generate_prompt(dp["system"], dp["input"], dp["output"])
    return _tokenize_train(
        tokenizer, full_prompt, args.llm_max_text_len, args.llm_train_on_inputs
    )


class LLMTrainDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, max_len, rng, text_dict, text_img_dict, tokenizer, prompter):
        self.args = args
        self.max_len = max_len
        self.num_items = args.num_items
        self.rng = rng
        self.text_dict = text_dict
        self.text_img_dict = text_img_dict
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.all_seqs = []
        for u in sorted(u2seq.keys()):
            seq = u2seq[u]
            for i in range(2, len(seq) + 1):
                self.all_seqs.append(seq[:i])

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        tokens = self.all_seqs[index]
        answer = tokens[-1]
        original_seq = tokens[:-1]
        seq = original_seq[-self.max_len :]
        candidates = [answer]
        samples = self.rng.randint(1, self.num_items + 1, size=5 * self.args.llm_negative_sample_size)
        cur = 0
        while len(candidates) < self.args.llm_negative_sample_size + 1:
            item = int(samples[cur])
            cur += 1
            if item in original_seq or item == answer:
                continue
            candidates.append(item)
        self.rng.shuffle(candidates)
        return seq_to_token_ids(
            self.args,
            seq,
            candidates,
            answer,
            self.text_dict,
            self.text_img_dict,
            self.tokenizer,
            self.prompter,
            eval_mode=False,
        )


class LLMEvalDataset(data_utils.Dataset):
    def __init__(
        self,
        args,
        u2seq,
        u2val,
        u2answer,
        max_len,
        text_dict,
        text_img_dict,
        tokenizer,
        prompter,
        users,
        candidates_list,
        *,
        include_val: bool,
    ):
        self.args = args
        self.u2seq = u2seq
        self.u2val = u2val
        self.u2answer = u2answer
        self.max_len = max_len
        self.text_dict = text_dict
        self.text_img_dict = text_img_dict
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.users = users
        self.candidates_list = candidates_list
        self.include_val = include_val

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        if self.include_val:
            seq = self.u2seq[user] + self.u2val[user]
        else:
            seq = self.u2seq[user]
        answer = self.u2answer[user][0]
        seq = seq[-self.max_len :]
        candidates = self.candidates_list[index]
        assert answer in candidates
        return seq_to_token_ids(
            self.args,
            seq,
            candidates,
            answer,
            self.text_dict,
            self.text_img_dict,
            self.tokenizer,
            self.prompter,
            eval_mode=True,
        )


def load_ranker_bundle(
    *,
    dataset_pkl: Path,
    retrieved_pkl: Path,
    tokenizer_name: str,
    llm_negative_sample_size: int = 19,
    llm_max_history: int = 25,
    llm_max_title_len: int = 32,
    llm_max_text_len: int = 1536,
    llm_train_on_inputs: bool = False,
    system_template: str | None = None,
    input_template: str | None = None,
    lora_micro_batch_size: int = 2,
    val_batch_size: int = 2,
    test_batch_size: int = 2,
    num_workers: int = 0,
    seed: int = 42,
    metric_ks: list[int] | None = None,
):
    from types import SimpleNamespace

    metric_ks = metric_ks or [1, 5, 10, 20, 50]
    dataset = load_pickle(dataset_pkl)
    if "meta_img_des" not in dataset:
        meta = dataset.get("meta") or {}
        dataset["meta_img_des"] = {int(k): "" for k in meta}
        logger.warning(
            "dataset.pkl missing meta_img_des — stubbed %s empty captions for text-only mode",
            len(meta),
        )
    retrieved = pickle.load(open(retrieved_pkl, "rb"))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    prompter = Prompter("alpaca_short")
    args = SimpleNamespace(
        num_items=len(dataset["smap"]),
        llm_max_history=llm_max_history,
        llm_max_title_len=llm_max_title_len,
        llm_max_text_len=llm_max_text_len,
        llm_train_on_inputs=llm_train_on_inputs,
        llm_negative_sample_size=llm_negative_sample_size,
        llm_system_template=system_template or DEFAULT_ML100K_SYSTEM,
        llm_input_template=input_template or DEFAULT_INPUT_TEMPLATE,
        metric_ks=metric_ks,
    )

    k = llm_negative_sample_size + 1
    val_probs, val_labels = retrieved["val_probs"], retrieved["val_labels"]
    test_probs, test_labels = retrieved["test_probs"], retrieved["test_labels"]
    test_metrics = retrieved["test_metrics"]

    val_users = [
        u
        for u, (p, l) in enumerate(zip(val_probs, val_labels), start=1)
        if l in torch.topk(torch.tensor(p), k).indices
    ]
    val_candidates = [
        torch.topk(torch.tensor(val_probs[u - 1]), k).indices.tolist() for u in val_users
    ]
    test_users = [
        u
        for u, (p, l) in enumerate(zip(test_probs, test_labels), start=1)
        if l in torch.topk(torch.tensor(p), k).indices
    ]
    test_candidates = [
        torch.topk(torch.tensor(test_probs[u - 1]), k).indices.tolist() for u in test_users
    ]
    non_test_users = [
        u
        for u, (p, l) in enumerate(zip(test_probs, test_labels), start=1)
        if l not in torch.topk(torch.tensor(p), k).indices
    ]
    test_retrieval = {
        "original_size": len(test_probs),
        "retrieval_size": len(test_candidates),
        "original_metrics": test_metrics,
        "retrieval_metrics": absolute_recall_mrr_ndcg_for_ks(
            torch.tensor(test_probs)[torch.tensor(test_users) - 1],
            torch.tensor(test_labels)[torch.tensor(test_users) - 1],
            metric_ks,
        )
        if test_users
        else {},
        "non_retrieval_metrics": absolute_recall_mrr_ndcg_for_ks(
            torch.tensor(test_probs)[torch.tensor(non_test_users) - 1],
            torch.tensor(test_labels)[torch.tensor(non_test_users) - 1],
            metric_ks,
        )
        if non_test_users
        else {},
    }
    logger.info(
        "ranker subsets: val_users=%s test_users=%s / %s",
        len(val_users),
        len(test_users),
        len(test_probs),
    )

    rng = np.random.RandomState(seed)
    train_ds = LLMTrainDataset(
        args,
        dataset["train"],
        llm_max_history,
        rng,
        dataset["meta"],
        dataset["meta_img_des"],
        tokenizer,
        prompter,
    )
    val_ds = LLMEvalDataset(
        args,
        dataset["train"],
        dataset["val"],
        dataset["val"],
        llm_max_history,
        dataset["meta"],
        dataset["meta_img_des"],
        tokenizer,
        prompter,
        val_users,
        val_candidates,
        include_val=False,
    )
    test_ds = LLMEvalDataset(
        args,
        dataset["train"],
        dataset["val"],
        dataset["test"],
        llm_max_history,
        dataset["meta"],
        dataset["meta_img_des"],
        tokenizer,
        prompter,
        test_users,
        test_candidates,
        include_val=True,
    )

    train_loader = data_utils.DataLoader(
        train_ds,
        batch_size=lora_micro_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: _collate_train(batch, tokenizer),
        worker_init_fn=worker_init_fn,
    )
    val_loader = data_utils.DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: _collate_eval(batch, tokenizer),
    )
    test_loader = data_utils.DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: _collate_eval(batch, tokenizer),
    )
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "tokenizer": tokenizer,
        "test_retrieval": test_retrieval,
        "args": args,
        "num_classes": k,
    }


def _pad_ids(seqs, pad_id: int):
    max_len = max(len(s) for s in seqs)
    out = []
    attn = []
    for s in seqs:
        pad = max_len - len(s)
        # left pad (official left padding)
        out.append([pad_id] * pad + s)
        attn.append([0] * pad + [1] * len(s))
    return torch.tensor(out), torch.tensor(attn)


def _collate_train(batch, tokenizer):
    ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    pad_id = tokenizer.pad_token_id
    input_ids, attention_mask = _pad_ids(ids, pad_id)
    # pad labels with -100
    max_len = input_ids.size(1)
    lab = []
    for row in labels:
        pad = max_len - len(row)
        lab.append([-100] * pad + row)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor(lab),
    }


def _collate_eval(batch, tokenizer):
    ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    input_ids, attention_mask = _pad_ids(ids, tokenizer.pad_token_id)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor(labels),
    }
