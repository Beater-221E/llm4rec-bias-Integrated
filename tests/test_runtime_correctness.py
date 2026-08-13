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
        "train": {"sft": {"global_batch_size": 64}, "rl": {"global_batch_size": 8}},
    }
    m = build_execution_manifest(cfg)
    assert "algorithm" in m and "execution" in m
    assert m["algorithm"]["sid"]["rqvae_seed"] == 2024
