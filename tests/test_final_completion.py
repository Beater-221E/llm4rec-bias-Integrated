"""Additional tests for final completion pass."""

from __future__ import annotations

import torch
import torch.nn as nn

from llm4rec.data.minionerec_sft import (
    build_sft_rows,
    fusion_seqrec_example,
    sid_item_feat_examples,
    sid_sft_example,
)
from llm4rec.trainers.ref_sync import maybe_sync_reference_model
from llm4rec.trainers.rollouts import ConstrainedBeamRollout
from llm4rec.trainers.schedulers import create_scheduler


class _FakeTable:
    levels = 3

    def __contains__(self, item):
        return True

    def sid(self, item):
        return f"<a_1><b_2><c_{item}>"

    def items(self):
        return ["10", "20"]

    def prefix_allowed_fn(self, tokenizer, prompt_len, eos):
        def _fn(batch_id, input_ids):
            return [1, 2, 3]

        return _fn


def test_minionerec_reproduction_sft_objectives():
    table = _FakeTable()
    meta = {"10": {"title": "Book A"}, "20": {"title": "Book B"}}
    rows = [{"user_id": "u1", "history": ["10"], "target_item": "20"}]
    exs = build_sft_rows(
        train_rows=rows,
        meta=meta,
        sid_table=table,
        objectives=["sid_sft", "sid_item_feat", "fusion_seqrec"],
    )
    tasks = {e["objective"] for e in exs}
    assert "sid_sft" in tasks
    assert "fusion_seqrec" in tasks
    assert "sid_item_feat" in tasks
    sid = sid_sft_example(user_id="u", history=["10"], target="20", sid_table=table)
    assert "chronological order" in sid["prompt"][0]["content"]
    assert sid["answer"].startswith("<a_1>")
    fusion = fusion_seqrec_example(
        user_id="u", history=["10"], target="20", sid_table=table, meta=meta
    )
    assert "Tell me the title" in fusion["prompt"][0]["content"]
    assert "Book B" in fusion["answer"]
    feats = sid_item_feat_examples(item="10", sid_table=table, meta=meta)
    assert len(feats) == 2


def test_constrained_rollout_defaults_do_sample_true():
    r = ConstrainedBeamRollout(_FakeTable())
    assert r.do_sample is True
    assert r.temperature == 1.0
    r2 = ConstrainedBeamRollout(_FakeTable(), do_sample=False)
    assert r2.do_sample is False


def test_scheduler_warmup_then_cosine():
    model = nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = create_scheduler(
        opt, scheduler_type="cosine", num_training_steps=20, warmup_steps=5
    )
    lrs = []
    for _ in range(20):
        opt.step()
        sched.step()
        lrs.append(sched.get_last_lr()[0])
    assert lrs[0] < lrs[4]  # warmup rising toward peak
    assert lrs[-1] < lrs[5]  # cosine decays after warmup


def test_ref_sync_interval_semantics():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(3))

    policy = Tiny()
    ref = Tiny()
    with torch.no_grad():
        policy.w.fill_(1.0)
        ref.w.fill_(0.0)
    assert maybe_sync_reference_model(policy, ref, 100, enabled=True, sync_steps=512) is False
    assert float(ref.w[0]) == 0.0
    assert maybe_sync_reference_model(policy, ref, 512, enabled=True, alpha=0.6, sync_steps=512)
    assert abs(float(ref.w[0]) - 0.6) < 1e-5
    ref2 = Tiny()
    with torch.no_grad():
        ref2.w.fill_(0.0)
    assert maybe_sync_reference_model(policy, ref2, 512, enabled=False, sync_steps=512) is False
    assert float(ref2.w[0]) == 0.0


def test_deepspeed_effective_falls_back_for_custom_wrap():
    from llm4rec.runtime.context import build_runtime

    cfg = {
        "mode": "reproduction",
        "experiment": {"route": "minionerec", "mode": "reproduction"},
        "hardware": {"precision": "fp32", "strategy": "zero2", "memory": "none", "deepspeed": None},
        "optimization": {"compile": {"enabled": False}},
    }
    rt = build_runtime(cfg, log=lambda *_: None)
    # Force resolved deepspeed then wrap
    rt.effective_strategy = "deepspeed_zero2"
    rt.resolved_strategy = "deepspeed_zero2"
    rt.strategy.strategy = "deepspeed_zero2"
    model = nn.Linear(2, 2)
    wrapped = rt.wrap_model(model)
    assert rt.effective_strategy == "single" or rt.effective_strategy == "ddp"
    assert "deepspeed" not in str(rt.effective_strategy)
    assert rt.fallback_reason == "custom_grpo_deepspeed_backend_not_implemented"
    assert wrapped is model or hasattr(wrapped, "module")


def test_fsdp_wrap_respects_fp32():
    from llm4rec.core import distributed as dist_utils

    # Single-process: wrap_fsdp returns model unchanged; just ensure API accepts dtypes
    m = nn.Linear(2, 2)
    out = dist_utils.wrap_fsdp(m, param_dtype=None, reduce_dtype=None, buffer_dtype=None)
    assert out is m


def test_manifest_has_reference_semantics():
    from llm4rec.runtime.manifest import build_execution_manifest

    cfg = {
        "mode": "reproduction",
        "seed": 42,
        "experiment": {"route": "minionerec"},
        "stages": ["sft", "rl"],
        "sid": {"implementation": "minionerec_reference", "rqvae": {"seed": 2024}},
        "hardware": {"precision": "bf16", "strategy": "ddp", "_hardware": {"device_count": 2, "gpu_name": "Test", "total_memory": 16 * 1024**3}},
        "train": {
            "sft": {"objectives": ["sid_sft", "sid_item_feat", "fusion_seqrec"], "learning_rate": 3e-4, "epochs": 10},
            "rl": {
                "optim": "paged_adamw_32bit",
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.03,
                "grpo": {"group_size": 16, "do_sample": True, "temperature": 1.0, "beam_search": True, "sync_ref_model": True},
            },
        },
        "optimization": {"generation": {"cache": "auto"}},
    }
    m = build_execution_manifest(cfg)
    assert m["reference_semantics"]["do_sample"] is True
    assert m["reference_semantics"]["sft_objectives"][0] == "sid_sft"
    assert "hardware" in m and "execution" in m
