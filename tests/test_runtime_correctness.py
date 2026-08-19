"""Parity + unit tests for B×G GRPO, 2B DPO, ref sync, batch policy."""

from __future__ import annotations

import torch
import torch.nn as nn

from llm4rec.trainers.dpo import dpo_loss
from llm4rec.trainers.grpo import batched_group_advantages, compute_grpo_loss, group_advantages
from llm4rec.trainers.logprobs import (
    batched_multi_prompt_logprobs,
    batched_pair_logprobs,
    batched_sequence_logprobs,
    logprob_chunk_size,
    score_preference_batch,
    sequence_logprobs,
)
from llm4rec.trainers.ref_sync import maybe_sync_reference_model


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 32, dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.out = nn.Linear(dim, vocab)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            input_ids = kwargs.get("input_ids")
        h = self.embed(input_ids)
        return type("O", (), {"logits": self.out(h)})()


def test_logprob_chunk_size_caps_sid_vocab():
    # 16 × 544 × 153201 fp32 ≈ 5.3 GiB; 512 MiB budget → chunk 1
    assert logprob_chunk_size(16, 544, 153201) == 1
    assert logprob_chunk_size(8, 64, 32) == 8


def test_chunked_logprobs_match_full_batch():
    torch.manual_seed(0)
    model = TinyLM()
    model.eval()
    prompts = [torch.randint(0, 32, (n,)) for n in (4, 5, 3, 6)]
    comps = [torch.randint(0, 32, (n,)) for n in (2, 6, 4, 3)]
    with torch.no_grad():
        full = batched_multi_prompt_logprobs(model, prompts, comps, pad_token_id=0)
        chunked = batched_multi_prompt_logprobs(
            model, prompts, comps, pad_token_id=0
        )
        # force 1-row chunks via private kw by going through _score_padded_batch
        from llm4rec.trainers.logprobs import _score_padded_batch

        forced = _score_padded_batch(
            model, prompts, comps, pad_token_id=0, max_chunk=1
        )
    for a, b, c in zip(full, chunked, forced):
        assert torch.allclose(a, b, atol=1e-5)
        assert torch.allclose(a, c, atol=1e-5)


def test_batched_multi_prompt_matches_sequential():
    torch.manual_seed(0)
    model = TinyLM()
    model.eval()
    prompts = [torch.randint(0, 32, (n,)) for n in (4, 5, 3)]
    comps = [torch.randint(0, 32, (n,)) for n in (2, 6, 4)]
    with torch.no_grad():
        seq = [sequence_logprobs(model, p, c) for p, c in zip(prompts, comps)]
        bat = batched_multi_prompt_logprobs(model, prompts, comps, pad_token_id=0)
    for a, b in zip(seq, bat):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_bxg_advantages_group_isolation():
    """Advantages for prompt A must not depend on prompt B rewards."""
    a = [1.0, 0.0, 0.5, 0.25]
    b = [10.0, 10.0, 10.0, 10.0]
    flat = batched_group_advantages([a, b], "group")
    only_a = group_advantages(a, "group")
    assert torch.allclose(flat[:4], only_a, atol=1e-6)
    # B has zero variance → zeros
    assert torch.allclose(flat[4:], torch.zeros(4), atol=1e-6)


def test_b1_vs_multi_prompt_scoring_parity():
    """B=1 path (shared prompt API) matches multi-prompt API for one prompt."""
    torch.manual_seed(3)
    model = TinyLM()
    prompt = torch.randint(0, 32, (5,))
    comps = [torch.randint(0, 32, (n,)) for n in (3, 4, 2)]
    with torch.no_grad():
        g1 = batched_sequence_logprobs(model, prompt, comps, pad_token_id=0)
        g2 = batched_multi_prompt_logprobs(
            model, [prompt] * len(comps), comps, pad_token_id=0
        )
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, atol=1e-5)


def test_score_preference_batch_matches_pairwise():
    torch.manual_seed(4)
    model = TinyLM()
    prompts = [torch.randint(0, 32, (4,)) for _ in range(3)]
    chosens = [torch.randint(0, 32, (5,)) for _ in range(3)]
    rejecteds = [torch.randint(0, 32, (6,)) for _ in range(3)]
    with torch.no_grad():
        pc_b, pr_b = score_preference_batch(
            model, prompts, chosens, rejecteds, pad_token_id=0
        )
        pcs, prs = [], []
        for p, c, r in zip(prompts, chosens, rejecteds):
            pc, pr = batched_pair_logprobs(model, p, c, r, pad_token_id=0)
            pcs.append(pc)
            prs.append(pr)
        pc_s = torch.stack(pcs)
        pr_s = torch.stack(prs)
    assert torch.allclose(pc_b, pc_s, atol=1e-5)
    assert torch.allclose(pr_b, pr_s, atol=1e-5)
    loss, _ = dpo_loss(pc_b, pr_b, pc_b.detach(), pr_b.detach(), beta=0.1)
    assert torch.isfinite(loss)


def test_sync_ref_model_mixup():
    torch.manual_seed(5)
    policy = TinyLM()
    ref = TinyLM()
    # Different init
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(1.0)
    before = [p.detach().clone() for p in ref.parameters()]
    did = maybe_sync_reference_model(
        policy, ref, step=512, enabled=True, alpha=0.6, sync_steps=512
    )
    assert did is True
    changed = any(not torch.equal(a, b) for a, b in zip(before, list(ref.parameters())))
    assert changed

    # Disabled: no change
    before2 = [p.detach().clone() for p in ref.parameters()]
    did2 = maybe_sync_reference_model(
        policy, ref, step=512, enabled=False, alpha=0.6, sync_steps=512
    )
    assert did2 is False
    assert all(torch.equal(a, b) for a, b in zip(before2, list(ref.parameters())))

    # Wrong step: no sync
    did3 = maybe_sync_reference_model(
        policy, ref, step=511, enabled=True, alpha=0.6, sync_steps=512
    )
    assert did3 is False


def test_batch_policy_best_effort_and_strict():
    from llm4rec.core.exceptions import ConfigurationError
    from llm4rec.runtime.batch import resolve_batch_plan
    import pytest

    p1 = resolve_batch_plan(
        world_size=1, per_device_batch_size=2, global_batch_size=64, mode="integrated"
    )
    p4 = resolve_batch_plan(
        world_size=4, per_device_batch_size=2, global_batch_size=64, mode="integrated"
    )
    assert p1.effective_global_batch_size == 64
    assert p4.effective_global_batch_size == 64
    assert p1.gradient_accumulation_steps == 32
    assert p4.gradient_accumulation_steps == 8

    soft = resolve_batch_plan(
        world_size=3,
        per_device_batch_size=2,
        global_batch_size=64,
        batch_policy={"preserve_global_batch": "best_effort", "max_relative_deviation": 0.05},
    )
    assert soft.adjusted
    assert soft.reference_global_batch == 64
    assert soft.relative_batch_deviation >= 0.0

    with pytest.raises(ConfigurationError):
        resolve_batch_plan(
            world_size=3,
            per_device_batch_size=2,
            global_batch_size=64,
            batch_policy={"preserve_global_batch": "strict", "max_relative_deviation": 0.0},
        )


def test_strategy_requested_resolved_effective():
    from llm4rec.runtime.context import build_runtime

    cfg = {
        "mode": "integrated",
        "experiment": {"route": "minionerec", "mode": "integrated"},
        "hardware": {"precision": "fp32", "strategy": "auto", "memory": "none"},
        "optimization": {"compile": {"enabled": False}},
        "train": {"rl": {"per_device_batch_size": 2, "global_batch_size": 8}},
    }
    rt = build_runtime(cfg, log=lambda *_: None)
    assert rt.requested_strategy == "auto"
    assert rt.resolved_strategy
    assert rt.effective_strategy
    # Tiny model params → stay on single/ddp depending on world
    model = TinyLM()
    rt.bind_model_params(model, log=lambda *_: None)
    assert rt.model_params_b is not None
    assert rt.model_params_b < 1.0


def test_execution_manifest_shape():
    from llm4rec.runtime.manifest import build_execution_manifest

    cfg = {
        "mode": "reproduction",
        "seed": 42,
        "experiment": {"route": "minionerec"},
        "stages": ["sft", "rl"],
        "sid": {"implementation": "minionerec_reference", "rqvae": {"seed": 2024}},
        "hardware": {
            "precision": "bf16",
            "strategy": "ddp",
            "batch_policy": {"preserve_global_batch": "best_effort"},
            "_resolved_strategy": {
                "requested_strategy": "auto",
                "resolved_strategy": "ddp",
                "effective_strategy": "ddp",
            },
        },
        "train": {
            "sft": {
                "global_batch_size": 64,
                "objectives": ["sid_sft", "sid_item_feat", "fusion_seqrec"],
            },
            "rl": {
                "global_batch_size": 8,
                "grpo": {"group_size": 16, "do_sample": True, "temperature": 1.0},
            },
        },
    }
    m = build_execution_manifest(cfg)
    assert "algorithm" in m and "execution" in m and "hardware" in m
    assert m["algorithm"]["sid"]["rqvae_seed"] == 2024
    refs = m.get("reference_semantics") or {}
    if refs:
        assert refs.get("do_sample") is True
        assert refs.get("sft_objectives", [None])[0] == "sid_sft"


def test_move_optimizer_state_cpu_and_back():
    from llm4rec.trainers.grpo import _move_optimizer_state

    model = nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(2, 4)).sum().backward()
    opt.step()
    _move_optimizer_state(opt, "cpu")
    for state in opt.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                assert value.device.type == "cpu"
    _move_optimizer_state(opt, torch.device("cpu"))


def test_generation_mode_restores_train_and_cache():
    from llm4rec.trainers.grpo import _generation_mode

    class _Cfg:
        use_cache = False

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)
            self.config = _Cfg()
            self.is_gradient_checkpointing = True
            self._gc = True

        def gradient_checkpointing_disable(self):
            self._gc = False

        def gradient_checkpointing_enable(self):
            self._gc = True

    model = _M()
    model.train()
    with _generation_mode(model) as core:
        assert core.training is False
        assert core.config.use_cache is True
        assert core._gc is False
    assert model.training is True
    assert model.config.use_cache is False
    assert model._gc is True


def test_scheduler_warmup_then_cosine():
    from llm4rec.trainers.schedulers import create_scheduler

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
    assert lrs[0] < lrs[4]
    assert lrs[-1] < lrs[5]


def test_deepspeed_effective_falls_back_for_custom_wrap():
    from llm4rec.runtime.context import build_runtime

    cfg = {
        "mode": "reproduction",
        "experiment": {"route": "minionerec", "mode": "reproduction"},
        "hardware": {
            "precision": "fp32",
            "strategy": "zero2",
            "memory": "none",
            "deepspeed": None,
        },
        "optimization": {"compile": {"enabled": False}},
    }
    rt = build_runtime(cfg, log=lambda *_: None)
    rt.effective_strategy = "deepspeed_zero2"
    rt.resolved_strategy = "deepspeed_zero2"
    rt.strategy.strategy = "deepspeed_zero2"
    model = nn.Linear(2, 2)
    wrapped = rt.wrap_model(model)
    assert "deepspeed" not in str(rt.effective_strategy)
    assert rt.fallback_reason == "custom_grpo_deepspeed_backend_not_implemented"
    assert wrapped is model or hasattr(wrapped, "module")


def test_fsdp_wrap_respects_fp32():
    from llm4rec.core import distributed as dist_utils

    m = nn.Linear(2, 2)
    out = dist_utils.wrap_fsdp(m, param_dtype=None, reduce_dtype=None, buffer_dtype=None)
    assert out is m
