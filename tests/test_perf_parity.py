"""Parity tests for batched logprobs and runtime wiring."""

from __future__ import annotations

import torch
import torch.nn as nn

from llm4rec.trainers.logprobs import batched_sequence_logprobs, sequence_logprobs
from llm4rec.trainers.grpo import compute_grpo_loss, group_advantages
from llm4rec.trainers.dpo import dpo_loss
from llm4rec.trainers.logprobs import batched_pair_logprobs


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 32, dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.out = nn.Linear(dim, vocab)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            input_ids = kwargs.get("input_ids")
        h = self.embed(input_ids)
        logits = self.out(h)
        return type("O", (), {"logits": logits})()


def test_batched_sequence_logprobs_matches_sequential():
    torch.manual_seed(0)
    model = TinyLM()
    model.eval()
    prompt = torch.randint(0, 32, (5,))
    comps = [torch.randint(0, 32, (n,)) for n in (3, 7, 2, 5)]
    with torch.no_grad():
        seq = [sequence_logprobs(model, prompt, c) for c in comps]
        bat = batched_sequence_logprobs(model, prompt, comps, pad_token_id=0)
    assert len(seq) == len(bat)
    for a, b in zip(seq, bat):
        assert a.shape == b.shape
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_batched_dpo_pair_matches_sequential():
    torch.manual_seed(1)
    model = TinyLM()
    prompt = torch.randint(0, 32, (4,))
    chosen = torch.randint(0, 32, (6,))
    rejected = torch.randint(0, 32, (5,))
    with torch.no_grad():
        pc = sequence_logprobs(model, prompt, chosen).sum()
        pr = sequence_logprobs(model, prompt, rejected).sum()
        bc, br = batched_pair_logprobs(model, prompt, chosen, rejected, pad_token_id=0)
    assert torch.allclose(pc, bc, atol=1e-5)
    assert torch.allclose(pr, br, atol=1e-5)


def test_compute_grpo_loss_finite():
    torch.manual_seed(2)
    logps = [torch.randn(4, requires_grad=True) for _ in range(4)]
    refs = [lp.detach() for lp in logps]
    adv = group_advantages([1.0, 0.0, 0.5, 0.2])
    loss, kls = compute_grpo_loss(logps, refs, adv, beta=0.001, clip_eps=0.2, kl_type="k1")
    assert torch.isfinite(loss)
    loss.backward()


def test_low_var_kl_nonnegative_when_advantages_zero():
    from llm4rec.trainers.grpo import low_var_kl

    logp = torch.tensor([-1.2, -0.4, -2.0], requires_grad=True)
    ref = torch.tensor([-1.0, -0.5, -1.5])
    kl = low_var_kl(logp, ref)
    assert float(kl) >= 0.0
    loss, kls = compute_grpo_loss(
        [logp], [ref], torch.zeros(1), beta=1e-3, clip_eps=0.2, kl_type="low_var_kl"
    )
    assert torch.isfinite(loss)
    assert kls[0] >= 0.0


def test_dpo_loss_finite():
    pc = torch.tensor([1.0])
    pr = torch.tensor([0.2])
    rc = torch.tensor([0.8])
    rr = torch.tensor([0.3])
    loss, stats = dpo_loss(pc, pr, rc, rr, beta=0.01)
    assert torch.isfinite(loss)
    assert "accuracy" in stats


def test_runtime_context_wires_precision_and_strategy():
    from llm4rec.runtime.context import build_runtime

    cfg = {
        "mode": "integrated",
        "experiment": {"route": "recr1", "mode": "integrated"},
        "hardware": {
            "precision": "auto",
            "strategy": "auto",
            "memory": "auto",
            "find_unused_parameters": False,
        },
        "optimization": {"compile": {"enabled": False}},
        "train": {"rl": {"per_device_batch_size": 1, "global_batch_size": 4}},
    }
    rt = build_runtime(cfg, log=lambda *_: None)
    assert rt.precision.precision in {"fp32", "fp16", "bf16"}
    assert rt.strategy.strategy in {"single", "ddp", "fsdp", "deepspeed_zero2", "deepspeed_zero3"}
    plan = rt.resolve_stage_batch("rl", cfg["train"]["rl"])
    assert plan.effective_global_batch_size == 4


def test_reproduction_config_forces_sid():
    from llm4rec.core.compose import compose, to_dict, validate

    cfg = validate(to_dict(compose("minionerec_reproduction_qwen05b")))
    assert cfg["mode"] == "reproduction"
    assert cfg["sid"]["implementation"] == "minionerec_reference"
    assert cfg["sid"]["rqvae"]["pca_dim"] is None
    assert cfg["sid"]["rqvae"]["num_emb_list"] == [256, 256, 256]
    assert cfg["sid"]["rqvae"]["e_dim"] == 32
    assert cfg["sid"]["rqvae"]["seed"] == 2024
    assert cfg["optimization"]["compile"]["enabled"] is False


def test_quantize_nearest_reference():
    from llm4rec.kernels import quantize_nearest

    torch.manual_seed(0)
    x = torch.randn(8, 16)
    cb = torch.randn(32, 16)
    idx = quantize_nearest(x, cb, backend="reference")
    assert idx.shape == (8,)
    # auto without triton flag uses reference
    idx2 = quantize_nearest(x, cb, backend="auto")
    assert torch.equal(idx, idx2)


def test_find_unused_parameters_default_false():
    from llm4rec.core import distributed as dist_utils
    import inspect

    sig = inspect.signature(dist_utils.wrap_ddp)
    assert sig.parameters["find_unused_parameters"].default is False
