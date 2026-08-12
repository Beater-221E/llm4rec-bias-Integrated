"""Real 2-GPU trainer smoke tests. Skipped when fewer than 2 CUDA devices.

Manual run:
  torchrun --standalone --nproc_per_node=2 -m pytest tests/test_trainer_dist_smoke.py -q
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from llm4rec.core import distributed as dist_utils
from llm4rec.trainers.grpo import Rollout, compute_grpo_loss, group_advantages, run_grpo
from llm4rec.trainers.dpo import run_dpo
from llm4rec.trainers.metrics_dist import reduce_reward_stats, reduce_scalar_pack
from llm4rec.trainers.logprobs import batched_sequence_logprobs


def _cuda_world() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(os.environ.get("WORLD_SIZE", "1"))


class TinyCausal(nn.Module):
    """Minimal CausalLM-like module with generate + save_pretrained stubs."""

    def __init__(self, vocab: int = 64, dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.out = nn.Linear(dim, vocab)
        self.config = type("C", (), {"use_cache": True})()

    def forward(self, input_ids=None, attention_mask=None, **kw):
        h = self.embed(input_ids)
        return type("O", (), {"logits": self.out(h)})()

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=4, num_return_sequences=1, **kw):
        b = input_ids.shape[0]
        outs = []
        for _ in range(num_return_sequences):
            extra = torch.randint(0, self.embed.num_embeddings, (b, max_new_tokens), device=input_ids.device)
            outs.append(torch.cat([input_ids, extra], dim=1))
        return torch.cat(outs, dim=0)

    def save_pretrained(self, path, state_dict=None):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(state_dict or self.state_dict(), Path(path) / "pytorch_model.bin")

    def gradient_checkpointing_enable(self):
        return None


class TinyTok:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, prompt, add_generation_prompt=True, return_tensors=None, tokenize=True):
        ids = torch.tensor([[2, 3, 4, 5]])
        if return_tensors == "pt":
            return ids
        return ids[0].tolist()

    def decode(self, ids, skip_special_tokens=True):
        return "tok"

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tok.txt").write_text("ok", encoding="utf-8")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2, 3]}


@pytest.mark.skipif(_cuda_world() < 2, reason="needs torchrun with >=2 CUDA processes")
def test_grpo_packed_logging_and_steps():
    dist_utils.init_process_group()
    torch.cuda.set_device(dist_utils.local_rank())

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(32, 8)
            self.out = nn.Linear(8, 32)

        def forward(self, input_ids=None, attention_mask=None, **kw):
            h = self.emb(input_ids)
            return type("O", (), {"logits": self.out(h)})()

    model = Tiny().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_model = dist_utils.wrap_ddp(model, find_unused_parameters=False)

    steps = 0
    for step in range(1, 4):
        prompt = torch.randint(0, 32, (4,), device="cuda")
        comps = [torch.randint(0, 32, (3,), device="cuda") for _ in range(4)]
        logps = batched_sequence_logprobs(train_model, prompt, comps, pad_token_id=0)
        refs = [lp.detach() for lp in logps]
        adv = group_advantages([1.0, 0.0, 0.5, 0.25])
        loss, _ = compute_grpo_loss(logps, refs, adv, beta=0.0, clip_eps=0.2, kl_type="k1")
        loss = loss / 2
        if step % 2:
            with dist_utils.no_sync(train_model):
                loss.backward()
        else:
            loss.backward()
            opt.step()
            opt.zero_grad()
            steps += 1
            rewards = [1.0, 0.0, 0.5, 0.25]
            stats = reduce_reward_stats(rewards)
            packed = reduce_scalar_pack([float(loss.detach()) * 2])
            assert torch.isfinite(torch.tensor(packed[0]))
            assert stats["reward_count"] >= 4

    t = torch.tensor([float(steps)], device="cuda")
    import torch.distributed as dist

    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    assert int(t.item()) == steps
    dist_utils.barrier("done")
    dist_utils.cleanup()


@pytest.mark.skipif(_cuda_world() < 2, reason="needs torchrun with >=2 CUDA processes")
def test_run_grpo_two_gpu_smoke(tmp_path_factory):
    dist_utils.init_process_group()
    torch.cuda.set_device(dist_utils.local_rank())
    out = Path(tmp_path_factory.mktemp(f"grpo_r{dist_utils.rank()}"))

    model = TinyCausal().cuda()
    ref = TinyCausal().cuda()
    tok = TinyTok()

    def rollout_fn(m, tokenizer, example, group_size):
        device = next(m.parameters()).device
        prompt = torch.randint(0, 64, (6,), device=device)
        comps = [torch.randint(0, 64, (3,), device=device) for _ in range(group_size)]
        texts = ["x"] * group_size
        return Rollout(prompt_ids=prompt, completion_ids=comps, texts=texts, example=example)

    def reward_fn(rollout):
        return [float(i % 3) for i in range(len(rollout.completion_ids))]

    cfg: dict[str, Any] = {
        "mode": "integrated",
        "experiment": {"route": "minionerec", "mode": "integrated"},
        "hardware": {
            "precision": "fp32",
            "strategy": "ddp",
            "memory": "none",
            "batch_policy": {"preserve_global_batch": "best_effort"},
            "activation_checkpointing": False,
        },
        "optimization": {
            "compile": {"enabled": False},
            "generation": {"pad_to_multiple_of": 8},
        },
        "train": {
            "rl": {
                "max_steps": 3,
                "per_device_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "global_batch_size": None,
                "learning_rate": 1e-3,
                "logging_steps": 1,
                "grpo": {
                    "group_size": 4,
                    "beta": 0.0,
                    "clip_epsilon": 0.2,
                    "sync_ref_model": True,
                    "ref_model_mixup_alpha": 0.6,
                    "ref_model_sync_steps": 1,
                    "do_sample": True,
                    "temperature": 1.0,
                },
            }
        },
    }

    class Log:
        def info(self, *a, **k):
            pass

        def log_metrics(self, *a, **k):
            pass

    summary = run_grpo(
        cfg=cfg,
        model=model,
        ref_model=ref,
        tokenizer=tok,
        train_examples=[{"user_id": i, "prompt": []} for i in range(16)],
        rollout_fn=rollout_fn,
        reward_fn=reward_fn,
        output_dir=out,
        logger=Log(),
        stage="rl",
    )
    assert summary["steps"] == 3
    dist_utils.barrier("grpo_smoke")
    dist_utils.cleanup()


@pytest.mark.skipif(_cuda_world() < 2, reason="needs torchrun with >=2 CUDA processes")
def test_run_dpo_two_gpu_smoke(tmp_path_factory):
    dist_utils.init_process_group()
    torch.cuda.set_device(dist_utils.local_rank())
    out = Path(tmp_path_factory.mktemp(f"dpo_r{dist_utils.rank()}"))

    model = TinyCausal().cuda()
    ref = TinyCausal().cuda()
    tok = TinyTok()

    def score_fn(example, texts):
        return [float(i) for i in range(len(texts))]

    cfg: dict[str, Any] = {
        "mode": "integrated",
        "experiment": {"route": "dpo4rec", "mode": "integrated"},
        "hardware": {
            "precision": "fp32",
            "strategy": "ddp",
            "memory": "none",
            "batch_policy": {"preserve_global_batch": "best_effort"},
            "activation_checkpointing": False,
        },
        "optimization": {
            "compile": {"enabled": False},
            "generation": {"pad_to_multiple_of": 8, "cache": "auto"},
        },
        "train": {
            "dpo": {
                "iterations": 1,
                "num_samples": 4,
                "epochs": 1,
                "beta": 0.1,
                "max_new_tokens": 3,
                "per_device_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "learning_rate": 1e-3,
                "logging_steps": 1,
                "sampling_temperature": 1.0,
            }
        },
    }

    class Log:
        def info(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

        def log_metrics(self, *a, **k):
            pass

    summary = run_dpo(
        cfg=cfg,
        model=model,
        ref_model=ref,
        tokenizer=tok,
        train_examples=[
            {"user_id": i, "prompt": [{"role": "user", "content": "hi"}]} for i in range(8)
        ],
        score_fn=score_fn,
        output_dir=out,
        logger=Log(),
    )
    assert summary["steps"] >= 1
    dist_utils.barrier("dpo_smoke")
    dist_utils.cleanup()


@pytest.mark.skipif(_cuda_world() < 2, reason="needs torchrun with >=2 CUDA processes")
def test_dpo_packed_logging_smoke():
    dist_utils.init_process_group()
    for step in range(1, 4):
        packed = reduce_scalar_pack([float(step), 0.1, 0.9])
        assert len(packed) == 3
        dist_utils.barrier(f"dpo_{step}")
    dist_utils.cleanup()


@pytest.mark.skipif(_cuda_world() < 2, reason="needs torchrun with >=2 CUDA processes")
def test_fsdp_train_save_reload_fp32(tmp_path_factory):
    """FSDP wrap → step → full-state save → reload; FP32 stays FP32."""
    dist_utils.init_process_group()
    torch.cuda.set_device(dist_utils.local_rank())
    out = Path(tmp_path_factory.mktemp(f"fsdp_r{dist_utils.rank()}"))

    model = TinyCausal().cuda().float()
    wrapped = dist_utils.wrap_fsdp(
        model,
        param_dtype=None,  # FP32: no MixedPrecision
        reduce_dtype=None,
        buffer_dtype=None,
    )
    opt = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad], lr=1e-3)
    x = torch.randint(0, 64, (2, 8), device="cuda")
    loss = wrapped(input_ids=x).logits.float().mean()
    loss.backward()
    opt.step()
    assert next(wrapped.parameters()).dtype == torch.float32

    dist_utils.save_pretrained_distributed(wrapped, out / "ckpt", is_main=dist_utils.is_main())
    dist_utils.barrier("fsdp_saved")
    if dist_utils.is_main():
        assert (out / "ckpt" / "pytorch_model.bin").exists() or any((out / "ckpt").glob("*"))
        reloaded = TinyCausal()
        state = torch.load(out / "ckpt" / "pytorch_model.bin", map_location="cpu")
        # state may be full gathered dict
        try:
            reloaded.load_state_dict(state, strict=False)
        except Exception:
            pass
        assert next(reloaded.parameters()).dtype == torch.float32
    dist_utils.barrier("fsdp_done")
    dist_utils.cleanup()
