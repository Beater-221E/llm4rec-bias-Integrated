"""Focused unit tests for SID, hardware, batch, and collective correctness."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch


# --------------------------------------------------------------------------- SID official


def test_minionerec_reference_rqvae_config():
    from llm4rec.sid.minionerec_rqvae import (
        MINIONEREC_RQVAE_DEFAULTS,
        MiniOneRecRQVAE,
        MiniOneRecRQVAEConfig,
        assert_minionerec_architecture,
    )

    cfg = MiniOneRecRQVAEConfig.from_dict({})
    assert_minionerec_architecture(cfg)
    assert cfg.layers == [2048, 1024, 512, 256, 128, 64]
    assert cfg.e_dim == 32
    assert cfg.num_emb_list == [256, 256, 256]
    assert MINIONEREC_RQVAE_DEFAULTS["pca_dim"] is None

    model = MiniOneRecRQVAE(in_dim=64, num_emb_list=[8, 8, 8], e_dim=8, layers=[32, 16])
    x = torch.randn(16, 64)
    out, rq_loss, indices = model(x, use_sk=False)
    assert out.shape == x.shape
    assert indices.shape[-1] == 3


def test_no_pca_in_minionerec_reproduction(tmp_path):
    from llm4rec.sid.minionerec_rqvae import train_minionerec_rqvae

    with pytest.raises(ValueError, match="must not use PCA"):
        train_minionerec_rqvae(
            np.random.randn(16, 32).astype(np.float32),
            {"pca_dim": 16, "epochs": 1, "batch_size": 8, "eval_step": 1},
            out_dir=tmp_path,
        )


def test_sid_collision_resolution():
    from llm4rec.sid.base import collision_rate, compute_collision_metrics
    from llm4rec.sid.minionerec_rqvae import MiniOneRecRQVAE, resolve_collisions_minionerec

    torch.manual_seed(0)
    model = MiniOneRecRQVAE(in_dim=16, num_emb_list=[8, 8, 8], e_dim=8, layers=[32, 16])
    # Force last-layer sinkhorn epsilon path
    x = torch.randn(20, 16)
    with torch.no_grad():
        codes = model.get_indices(x, use_sk=False).cpu().numpy()
    raw = collision_rate(codes)
    resolved = resolve_collisions_minionerec(
        model, x, codes, sk_epsilon=0.003, max_iters=5, device="cpu", log=lambda *_: None
    )
    metrics = compute_collision_metrics(resolved, raw_codes=codes)
    assert metrics.raw_collision_rate == pytest.approx(raw)
    assert "post_resolution_collision_rate" in metrics.to_dict()
    # Resolution should not increase collisions
    assert metrics.post_resolution_collision_rate <= raw + 1e-9


def test_integrated_sinkhorn_small_batch_large_codebook():
    """Regression: collision groups have B << K (e.g. 3 vs 512)."""
    from llm4rec.sid.collision import resolve_collisions_sinkhorn
    from llm4rec.sid.rqvae import RQVAE, sinkhorn

    torch.manual_seed(0)
    # B=3, K=512 — the shape that previously crashed via broadcast
    logits = -torch.rand(3, 512)
    q = sinkhorn(logits, epsilon=0.003, iters=5)
    assert q.shape == (3, 512)
    assert torch.isfinite(q).all()

    model = RQVAE(in_dim=16, latent_dim=8, hidden_dim=16, num_layers=3, codebook_size=64)
    x = torch.randn(24, 16)
    with torch.no_grad():
        codes = model.encode_indices(x, use_sk=False).cpu().numpy()
    # Force at least one collision so Sinkhorn path runs
    codes[1] = codes[0]
    resolved = resolve_collisions_sinkhorn(
        model, x, codes, sk_epsilon=0.003, max_iters=3, device="cpu", log=lambda *_: None
    )
    assert resolved.shape == codes.shape


def _write_sid_table(path: Path, mapping: dict[str, list[int]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    levels = len(next(iter(mapping.values())))
    codebook = max(max(c) for c in mapping.values()) + 1
    sid_map = {
        item: {"codes": codes, "sid": "".join(f"<{p}_{c}>" for p, c in zip("abc", codes))}
        for item, codes in mapping.items()
    }
    (path / "item2sid.json").write_text(json.dumps(sid_map), encoding="utf-8")
    manifest = {
        "config_hash": "test",
        "dataset": "toy",
        "seed": 0,
        "items_fingerprint": "x",
        "method": "rqvae",
        "levels": levels,
        "codebook_size": codebook,
        "layer_prefixes": ["a", "b", "c"][:levels],
        "n_items": len(mapping),
        "collision_rate": 0.0,
        "encoder": "",
        "created_at": "",
        "sid_config": {},
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_sid_table_preserves_collisions(tmp_path):
    from llm4rec.sid.table import SidTable

    # no collision
    d0 = tmp_path / "none"
    _write_sid_table(d0, {"1": [0, 0, 0], "2": [0, 0, 1]})
    t0 = SidTable(d0)
    assert t0.sid_to_items[(0, 0, 0)] == ["1"]
    assert t0.parse("<a_0><b_0><c_0>") == "1"

    # two items sharing one SID — both preserved; deterministic decode picks smaller id
    d1 = tmp_path / "col"
    _write_sid_table(d1, {"b": [1, 2, 3], "a": [1, 2, 3], "c": [9, 9, 9]})
    t1 = SidTable(d1)
    assert t1.sid_to_items[(1, 2, 3)] == ["a", "b"]
    assert t1.resolve_item([1, 2, 3]) == "a"
    assert t1.parse("<a_1><b_2><c_3>") == "a"
    assert t1.parse_all("<a_1><b_2><c_3>") == ["a", "b"]
    summary = t1.collision_summary()
    assert summary["num_collision_groups"] == 1
    assert summary["max_collision_group_size"] == 2

    # duplicate metadata → same SID list order still deterministic
    assert t1.items_for_sid([1, 2, 3])[0] == "a"


# --------------------------------------------------------------------------- runtime


def test_precision_resolver():
    from llm4rec.runtime.hardware import HardwareInfo
    from llm4rec.runtime.precision import resolve_precision

    cpu = HardwareInfo(
        device_count=0,
        local_rank=0,
        world_size=1,
        gpu_name="cpu",
        compute_capability=None,
        total_memory=None,
        free_memory=None,
        bf16_supported=False,
        tf32_supported=False,
        distributed=False,
        cuda_available=False,
    )
    assert resolve_precision("auto", cpu).precision == "fp32"

    ampere = HardwareInfo(
        device_count=1,
        local_rank=0,
        world_size=1,
        gpu_name="NVIDIA A100",
        compute_capability=(8, 0),
        total_memory=40 << 30,
        free_memory=35 << 30,
        bf16_supported=True,
        tf32_supported=True,
        distributed=False,
        cuda_available=True,
    )
    assert resolve_precision("auto", ampere).precision == "bf16"

    v100 = HardwareInfo(
        device_count=1,
        local_rank=0,
        world_size=1,
        gpu_name="Tesla V100",
        compute_capability=(7, 0),
        total_memory=32 << 30,
        free_memory=30 << 30,
        bf16_supported=False,
        tf32_supported=False,
        distributed=False,
        cuda_available=True,
    )
    # MiniOneRec defaults to fp32 on non-bf16 for SID embedding stability
    assert resolve_precision("auto", v100, route="minionerec").precision == "fp32"
    assert resolve_precision("auto", v100, route="recr1").precision == "fp16"
    from llm4rec.runtime.precision import weight_dtype_name

    fp16_amp = resolve_precision("auto", v100, route="dpo4rec")
    assert fp16_amp.grad_scaler is True
    assert weight_dtype_name(fp16_amp, trainable=True) == "fp32"
    assert weight_dtype_name(fp16_amp, trainable=False) == "fp16"
    with pytest.raises(ValueError, match="FP8"):
        resolve_precision("fp8", ampere)


def test_hardware_capability_detection():
    from llm4rec.runtime.hardware import detect_hardware

    hw = detect_hardware()
    assert hw.world_size >= 1
    assert isinstance(hw.bf16_supported, bool)
    assert isinstance(hw.cuda_available, bool)
    d = hw.to_dict()
    assert "gpu_name" in d


def test_global_batch_single_vs_multi_gpu():
    from llm4rec.runtime.batch import resolve_batch_plan

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


# --------------------------------------------------------------------------- checkpoint tokenizer rebuild


def test_has_tokenizer_files(tmp_path):
    from llm4rec.sid.model import _has_tokenizer_files

    assert not _has_tokenizer_files(tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert not _has_tokenizer_files(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert _has_tokenizer_files(tmp_path)


def test_load_tokenizer_rebuilds_stub_checkpoint(tmp_path, monkeypatch):
    from llm4rec.sid import model as m

    class FakeTok:
        def __init__(self, source: str):
            self.source = source
            self.chat_template = "TPL" if source == "backbone" else None
            self.pad_token = None
            self.eos_token = "</s>"
            self._tokens = ["<pad>"] * (151665 if source == "backbone" else 1)

        def __len__(self) -> int:
            return len(self._tokens)

        def add_tokens(self, tokens, special_tokens=False):
            before = len(self._tokens)
            self._tokens.extend(list(tokens))
            return len(self._tokens) - before

    def fake_from_pretrained(src, **_kw):
        return FakeTok("backbone" if str(src) == "bb" else "ckpt")

    monkeypatch.setattr(m.AutoTokenizer, "from_pretrained", staticmethod(fake_from_pretrained))

    class Table:
        def all_tokens(self):
            return [f"<a_{i}>" for i in range(8)]

    tok = m._load_tokenizer_for_checkpoint(tmp_path, backbone="bb", sid_table=Table())
    assert tok.source == "backbone"
    assert len(tok) == 151665 + 8
    assert tok.chat_template == "TPL"
    assert tok.pad_token == "</s>"


def test_qwen_embedding_padding_is_not_sid_mismatch():
    """Qwen2.5: tokenizer 151665, embedding 151936 — padding, not missing SID."""
    from llm4rec.sid.model import _assert_tokenizer_matches_model

    class Tok:
        unk_token_id = 0

        def __len__(self):
            return 151665

    class Emb:
        num_embeddings = 151936

    class Model:
        def get_input_embeddings(self):
            return Emb()

    _assert_tokenizer_matches_model(Tok(), Model())


def test_tokenizer_larger_than_embedding_still_errors():
    from llm4rec.core.exceptions import ConfigurationError
    from llm4rec.sid.model import _assert_tokenizer_matches_model

    class Tok:
        def __len__(self):
            return 200

    class Emb:
        num_embeddings = 100

    class Model:
        def get_input_embeddings(self):
            return Emb()

    with pytest.raises(ConfigurationError, match="大于模型 embedding"):
        _assert_tokenizer_matches_model(Tok(), Model())


def test_encode_prompt_rejects_empty_ids():
    from llm4rec.decoders.constrained_beam import _encode_prompt

    class EmptyTok:
        def __call__(self, text, return_tensors="pt", add_special_tokens=True):
            return {"input_ids": torch.zeros((1, 0), dtype=torch.long)}

        def apply_chat_template(self, *_a, **_k):
            return ""

    with pytest.raises(RuntimeError, match="empty input_ids"):
        _encode_prompt(EmptyTok(), [{"role": "user", "content": "hi"}])


# --------------------------------------------------------------------------- collectives


def test_grpo_collectives_all_ranks():
    """Static check: logging collectives use identical should_log on all ranks."""
    src = Path("src/llm4rec/trainers/grpo.py").read_text(encoding="utf-8")
    assert "should_log = state.step % logging_steps == 0" in src
    assert "reduce_reward_stats" in src
    # Collectives for metrics are inside should_log so all ranks agree on the gate
    assert "if should_log:" in src
    gate = src.index("if should_log:")
    assert src.index("reduce_reward_stats", gate) > gate
    assert "if is_main:" in src[gate:]


def test_dpo_collectives_all_ranks():
    src = Path("src/llm4rec/trainers/dpo.py").read_text(encoding="utf-8")
    assert "should_log = global_step % logging_steps == 0" in src
    assert "reduce_scalar_pack" in src
    assert src.index("if should_log:") < src.index("if is_main:")
    assert "_align_iteration_state" in src
    assert src.index("_align_iteration_state") < src.index("开始训练")


def test_align_dpo_iteration_state_single_process():
    from llm4rec.trainers.dpo import _align_iteration_state

    class Log:
        def info(self, *a, **k):
            pass

    pairs = [{"id": 1}, {"id": 2}, {"id": 3}]
    aligned, merged, stop = _align_iteration_state(
        pairs, {"u1": "reason"}, Log()
    )
    assert stop is False
    assert aligned == pairs
    assert merged == {"u1": "reason"}

    empty, merged_empty, stop_empty = _align_iteration_state([], {}, Log())
    assert stop_empty is True
    assert empty == []
    assert merged_empty == {}


def test_all_reduce_mean_single_process():
    from llm4rec.core.distributed import all_reduce_mean, all_reduce_min_int

    assert all_reduce_mean(3.0) == 3.0
    assert all_reduce_min_int(7) == 7


# --------------------------------------------------------------------------- mode / smoke control flow


def test_mode_defaults_minionerec_reproduction():
    from llm4rec.core.modes import apply_mode_defaults

    cfg = {
        "mode": "reproduction",
        "experiment": {"name": "t", "route": "minionerec"},
        "sid": {"method": "rqvae", "max_collision_rate": 0.0},
        "train": {},
        "hardware": {},
        "stages": ["sft", "eval", "rl", "eval"],
    }
    out = apply_mode_defaults(cfg)
    assert out["sid"]["implementation"] == "minionerec_reference"
    assert out["sid"]["rqvae"]["pca_dim"] is None
    assert out["sid"]["rqvae"]["layers"] == [2048, 1024, 512, 256, 128, 64]
    assert out["sid"]["max_collision_rate"] == 1.0


def test_smoke_control_flow_sid_official_tiny(tmp_path):
    """Tiny end-to-end official SID train+resolve (CPU, few epochs)."""
    from llm4rec.sid.minionerec_rqvae import (
        MiniOneRecRQVAEConfig,
        resolve_collisions_minionerec,
        train_minionerec_rqvae,
    )

    feats = np.random.randn(24, 32).astype(np.float32)
    cfg = MiniOneRecRQVAEConfig.from_dict(
        {
            "epochs": 3,
            "batch_size": 8,
            "eval_step": 1,
            "layers": [64, 32],
            "e_dim": 8,
            "num_emb_list": [8, 8, 8],
            "warmup_epochs": 0,
        }
    )
    # Not asserting official arch here — smoke only
    model, codes = train_minionerec_rqvae(
        feats, cfg, seed=0, device="cpu", out_dir=tmp_path, log=lambda *_: None
    )
    assert codes.shape == (24, 3)
    resolved = resolve_collisions_minionerec(
        model, feats, codes, max_iters=2, device="cpu", log=lambda *_: None
    )
    assert resolved.shape == codes.shape


def test_shared_run_stamp_uses_env(monkeypatch, tmp_path):
    from llm4rec.cli.main import _shared_run_stamp

    monkeypatch.setenv("LLM4REC_RUN_TS", "20260816_101517")
    assert _shared_run_stamp(tmp_path) == "20260816_101517"


def test_shared_run_stamp_rendezvous(monkeypatch, tmp_path):
    from llm4rec.cli.main import _shared_run_stamp

    monkeypatch.delenv("LLM4REC_RUN_TS", raising=False)
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("RANK", "0")
    a = _shared_run_stamp(tmp_path)
    monkeypatch.setenv("RANK", "1")
    b = _shared_run_stamp(tmp_path)
    assert a == b
    assert (tmp_path / ".run_stamp_29500").read_text().strip() == a
