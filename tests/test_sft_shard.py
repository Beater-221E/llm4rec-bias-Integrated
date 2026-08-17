"""SFT shard + allreduce path (no DDP), matching MiniOneRec GRPO."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm4rec.core import distributed as dist_utils
from llm4rec.trainers.sft import (
    ChatSFTDataset,
    PadCollator,
    _run_sft_sharded,
    _shard_dataset,
)


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 32, dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.lm_head = nn.Linear(dim, vocab)
        self.config = SimpleNamespace(use_cache=True, vocab_size=vocab)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        h = self.embed(input_ids)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)

    def save_pretrained(self, path, state_dict=None):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(state_dict or self.state_dict(), Path(path) / "pytorch_model.bin")

    def gradient_checkpointing_enable(self):
        return None


class _TinyTok:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, prompt, add_generation_prompt=True, tokenize=True):
        return [2, 3, 4]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [5, 6]}

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


class _RT:
    precision = SimpleNamespace(precision="fp32", grad_scaler=False)
    effective_strategy = "single"
    strategy = SimpleNamespace(strategy="single", pressure_ratio=None, source=None)
    world_size = 1
    mode = "integrated"
    scaler = None

    def maybe_compile(self, model, name=""):
        return model

    def wrap_model(self, model):
        raise AssertionError("sharded SFT must not wrap DDP")

    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def optimizer_step(self, optimizer, *, parameters, max_grad_norm):
        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        optimizer.step()


class _Log:
    def info(self, msg):
        return None

    def log_metrics(self, *args, **kwargs):
        return None


def test_shard_dataset_is_noop_without_dist():
    ds = list(range(8))

    class _ListDS:
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    out = _shard_dataset(_ListDS(ds))
    assert len(out) == 8
    assert [out[i] for i in range(8)] == ds


def test_shard_strided_matches_dist_utils():
    items = list(range(10))
    assert dist_utils.shard(items) == items


def test_sft_sharded_loop_cpu(tmp_path):
    tok = _TinyTok()
    examples = [
        {"prompt": [{"role": "user", "content": "a"}], "answer": "b"}
        for _ in range(8)
    ]
    ds = ChatSFTDataset(examples, tok, max_length=16)
    model = _TinyLM()
    cfg = {
        "train": {
            "sft": {
                "max_steps": 2,
                "epochs": 1,
                "logging_steps": 1,
                "learning_rate": 1e-3,
                "lr_scheduler_type": "constant",
                "per_device_batch_size": 2,
                "gradient_accumulation_steps": 1,
            }
        },
        "checkpoint": {"save_steps": None},
        "profiling": {"enabled": False},
    }
    plan = SimpleNamespace(
        per_device_batch_size=2,
        gradient_accumulation_steps=1,
        to_dict=lambda: {"per_device_batch_size": 2},
    )
    summary = _run_sft_sharded(
        cfg=cfg,
        sft_cfg=cfg["train"]["sft"],
        model=model,
        tokenizer=tok,
        train_ds=ds,
        eval_ds=None,
        n_train_global=len(examples),
        n_eval_global=0,
        output_dir=tmp_path,
        logger=_Log(),
        stage="sft",
        runtime=_RT(),
        batch_plan=plan,
        collator=PadCollator(0),
        precision="fp32",
        reference_sft=False,
    )
    assert summary["parallel"] == "shard_allreduce"
    assert summary["n_train"] == 8
    assert (tmp_path / "final" / "pytorch_model.bin").is_file()
    assert summary["metrics"]["train_steps"] == 2
