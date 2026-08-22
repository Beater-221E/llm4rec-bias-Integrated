"""Transition teacher + SID distill: mapping, numerics, pipeline, GRPO regression."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm4rec.core.compose import compose, to_dict, validate
from llm4rec.core.exceptions import ConfigurationError
from llm4rec.sid.table import SidTable, sid_token
from llm4rec.sid.transition import (
    SidTransitionTeacher,
    TransitionModel,
    run_transition,
    semantic_level_count,
    windows_from_examples,
)
from llm4rec.trainers.sid_distill import (
    distill_total_loss,
    exposure_alignment_loss,
    hard_sid_loss,
    level1_student_probs,
    soft_sid_loss,
)


# ------------------------------------------------------------------ fixtures


def _write_sid_table(
    path: Path,
    mapping: dict[str, list[int]],
    *,
    codebook_size: int | None = None,
    sid_config: dict | None = None,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    levels = len(next(iter(mapping.values())))
    observed = max(max(c) for c in mapping.values()) + 1
    codebook = int(codebook_size or observed)
    prefixes = ["a", "b", "c", "d", "e"][:levels]
    sid_map = {
        item: {
            "codes": codes,
            "sid": "".join(f"<{p}_{c}>" for p, c in zip(prefixes, codes)),
        }
        for item, codes in mapping.items()
    }
    (path / "item2sid.json").write_text(json.dumps(sid_map), encoding="utf-8")
    manifest = {
        "config_hash": "test-sda",
        "dataset": "toy",
        "seed": 0,
        "items_fingerprint": "toy-fp",
        "method": "rqvae",
        "levels": levels,
        "codebook_size": codebook,
        "layer_prefixes": prefixes,
        "n_items": len(mapping),
        "collision_rate": 0.0,
        "encoder": "",
        "created_at": "",
        "sid_config": sid_config or {},
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


class _Log:
    def info(self, *a, **k):
        return None

    def warning(self, *a, **k):
        return None

    def log_metrics(self, *a, **k):
        return None


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 64, dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.lm_head = nn.Linear(dim, vocab)
        self.config = SimpleNamespace(use_cache=True, vocab_size=vocab)

    def forward(self, input_ids=None, attention_mask=None, logits_to_keep=None, **kw):
        h = self.embed(input_ids)
        logits = self.lm_head(h)
        if logits_to_keep is not None:
            keep = int(logits_to_keep)
            logits = logits[:, -keep:, :]
        return SimpleNamespace(logits=logits)

    def get_output_embeddings(self):
        return self.lm_head

    def save_pretrained(self, path, state_dict=None):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(state_dict or self.state_dict(), Path(path) / "pytorch_model.bin")

    def gradient_checkpointing_enable(self):
        return None


class _TinyTok:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self._ids = {"<pad>": 0, "</s>": 1}

    def apply_chat_template(self, prompt, add_generation_prompt=True, tokenize=True):
        return [2, 3, 4, 5]

    def convert_tokens_to_ids(self, token):
        if token not in self._ids:
            self._ids[token] = len(self._ids)
        return self._ids[token]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2, 3]}

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
        return model

    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def optimizer_step(self, optimizer, parameters, max_grad_norm):
        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        optimizer.step()

    def bind_model_params(self, model, stage="sft", log=print, has_reference_model=None):
        return 0.0

    def resolve_stage_batch(self, stage, block):
        from llm4rec.runtime.batch import resolve_batch_plan

        return resolve_batch_plan(
            world_size=1,
            per_device_batch_size=int(block.get("per_device_batch_size") or 1),
            gradient_accumulation_steps=int(block.get("gradient_accumulation_steps") or 1),
            global_batch_size=block.get("global_batch_size"),
            mode="integrated",
        )


def _catalog(item_ids: list[str], counts: dict[str, int] | None = None):
    return SimpleNamespace(
        item_ids=list(item_ids),
        counts={i: int((counts or {}).get(i, 1)) for i in item_ids},
    )


def _prompt_row(history: list[str], target: str) -> dict:
    return {
        "history": history,
        "target_item": target,
        "prompt": [
            {"role": "system", "content": "sid"},
            {"role": "user", "content": "next"},
        ],
    }


# ------------------------------------------------------------------ 1 / 2 Transition numerics


def test_transition_forward_backward_heterogeneous_codebooks():
    torch.manual_seed(0)
    model = TransitionModel(codebook_sizes=[4, 8, 3], embedding_dim=16, hidden_dim=32, dropout=0.0)
    codes = torch.tensor(
        [
            [[0, 1, 0], [1, 7, 2], [-1, -1, -1]],
            [[3, 0, 1], [2, 4, 2], [1, 2, 0]],
        ],
        dtype=torch.long,
    )
    target = torch.tensor([[2, 3, 1], [0, 7, 2]], dtype=torch.long)
    loss = model(codes, target)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_joint_nll_matches_manual_sum():
    torch.manual_seed(1)
    model = TransitionModel(codebook_sizes=[5, 4], embedding_dim=8, hidden_dim=16, dropout=0.0)
    model.eval()
    codes = torch.tensor([[[0, 1], [2, 3], [-1, -1]]], dtype=torch.long)
    target = torch.tensor([[1, 2]], dtype=torch.long)
    with torch.no_grad():
        hidden = model.encode(codes)
        manual = torch.zeros(1)
        for layer in range(2):
            logits = model.level_logits(hidden, [target[:, j] for j in range(layer)])
            manual = manual + F.cross_entropy(logits, target[:, layer], reduction="none")
        got = model.joint_nll(codes, target)
        mean = model(codes, target)
    assert torch.allclose(got, manual, atol=1e-6)
    assert torch.allclose(mean, got.mean(), atol=1e-6)


def test_teacher_catalog_normalized(tmp_path):
    mapping = {
        "i0": [0, 0],
        "i1": [0, 1],
        "i2": [1, 0],
        "i3": [1, 1],
        "i4": [2, 0],
    }
    table = SidTable(_write_sid_table(tmp_path / "sid", mapping))
    model = TransitionModel(table.level_codebook_sizes(), embedding_dim=8, hidden_dim=16, dropout=0.0)
    teacher = SidTransitionTeacher(
        model,
        table,
        catalog=_catalog(list(mapping)),
        history_max_length=4,
        device="cpu",
    )
    logp = teacher.log_p_all([["i0", "i1"]])
    z = torch.logsumexp(logp, dim=-1)
    assert logp.shape == (1, 5)
    assert torch.allclose(z, torch.zeros_like(z), atol=1e-5)


def test_collision_items_are_not_merged(tmp_path):
    # Same full SID → mass split, two catalog columns remain.
    mapping = {"a": [0, 0, 0], "b": [0, 0, 0], "c": [1, 0, 0]}
    table = SidTable(_write_sid_table(tmp_path / "col", mapping))
    model = TransitionModel(table.level_codebook_sizes(), embedding_dim=8, hidden_dim=16, dropout=0.0)
    teacher = SidTransitionTeacher(
        model, table, catalog=_catalog(list(mapping)), history_max_length=3, device="cpu"
    )
    logp = teacher.log_p_all([["c"]])
    assert logp.shape == (1, 3)
    p = logp.exp()
    assert torch.allclose(p[0, 0], p[0, 1], atol=1e-5)
    assert float(p[0, 0] + p[0, 1] + p[0, 2]) == pytest.approx(1.0, abs=1e-5)

    # Distinct last-layer codes stay distinct (not collapsed to a prefix).
    mapping2 = {"x": [0, 0, 0], "y": [0, 0, 1], "z": [1, 0, 0]}
    table2 = SidTable(_write_sid_table(tmp_path / "dist", mapping2))
    model2 = TransitionModel(table2.level_codebook_sizes(), embedding_dim=8, hidden_dim=16, dropout=0.0)
    teacher2 = SidTransitionTeacher(
        model2, table2, catalog=_catalog(list(mapping2)), history_max_length=3, device="cpu"
    )
    logp2 = teacher2.log_p_all([["z"]])
    assert logp2.shape[1] == 3
    assert not torch.allclose(logp2[0, 0], logp2[0, 1])


def test_semantic_levels_drop_last_only_when_marked(tmp_path):
    mapping = {"1": [0, 0, 0], "2": [1, 2, 3]}
    plain = SidTable(_write_sid_table(tmp_path / "plain", mapping))
    marked = SidTable(
        _write_sid_table(
            tmp_path / "marked",
            mapping,
            sid_config={"last_level_collision_only": True},
        )
    )
    assert semantic_level_count(plain, {}) == 3
    assert semantic_level_count(marked, {}) == 2
    # sinkhorn_last_level is a build algorithm, not a drop-last flag
    assert semantic_level_count(plain, {"collision_handling": "sinkhorn_last_level"}) == 3


def test_windows_from_examples_use_unified_fields(tmp_path):
    table = SidTable(
        _write_sid_table(tmp_path / "w", {"a": [0, 0], "b": [1, 1], "c": [2, 2]})
    )
    pairs = windows_from_examples(
        [
            {"history": ["a", "b"], "target_item": "c"},
            {"history": [], "target_item": "c"},
            {"history": ["a"], "target_item": "missing"},
        ],
        table,
        history_max_length=10,
    )
    assert pairs == [(["a", "b"], "c")]


# ------------------------------------------------------------------ 3 / 4 sampling + collator


def test_distill_sampling_follows_teacher(tmp_path):
    mapping = {f"i{k}": [k, 0] for k in range(4)}
    table = SidTable(_write_sid_table(tmp_path / "s", mapping))
    model = TransitionModel(table.level_codebook_sizes(), embedding_dim=8, hidden_dim=16, dropout=0.0)
    teacher = SidTransitionTeacher(
        model, table, catalog=_catalog(list(mapping)), history_max_length=3, device="cpu"
    )
    hist = [["i0"]]
    logp = teacher.log_p_all(hist)
    p = torch.exp(logp - logp.max())
    p = (p / p.sum()).squeeze(0)
    gen = torch.Generator().manual_seed(0)
    sampled, _, _ = teacher.sample_items(hist, n_samples=4000, generator=gen)
    counts = torch.zeros(4)
    for item in sampled[0]:
        counts[teacher.item_pos[item]] += 1
    freq = counts / counts.sum()
    # Monte Carlo; 4000 draws, allow a loose TV distance
    assert float((freq - p).abs().sum()) < 0.12


def test_collator_masks_prompt_keeps_sid_completion(tmp_path):
    from llm4rec.data.minionerec_distill import MiniOneRecDistillCollator, MiniOneRecDistillDataset

    mapping = {"h": [0, 1], "t": [1, 0], "s": [2, 1]}
    table = SidTable(_write_sid_table(tmp_path / "c", mapping))
    model = TransitionModel(table.level_codebook_sizes(), embedding_dim=8, hidden_dim=16, dropout=0.0)
    teacher = SidTransitionTeacher(
        model, table, catalog=_catalog(list(mapping)), history_max_length=3, device="cpu"
    )
    tok = _TinyTok()
    ds = MiniOneRecDistillDataset([_prompt_row(["h"], "t")], table)
    assert list(ds[0].keys()) == ["prompt", "history", "target_item"]
    collator = MiniOneRecDistillCollator(tok, table, teacher, samples_per_prompt=3)
    batch = collator([ds[0]])
    assert batch["n_prompts"] == 1
    assert int(batch["gold_mask"].sum()) == 1
    assert batch["gold_mask"][-1]
    assert torch.isclose(batch["soft_weight"][~batch["gold_mask"]].sum(), torch.tensor(1.0))
    prompt_ref = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=True)
    gold_sid = [
        tok.convert_tokens_to_ids(sid_token(0, table.codes["t"][0], table.prefixes)),
        tok.convert_tokens_to_ids(sid_token(1, table.codes["t"][1], table.prefixes)),
    ]
    for p, c, n, is_gold in zip(
        batch["prompt_ids"],
        batch["completion_ids"],
        batch["n_sid_tokens"],
        batch["gold_mask"].tolist(),
    ):
        assert p.tolist() == prompt_ref
        assert int(c.numel()) == int(n) == 2
        if is_gold:
            assert c.tolist() == gold_sid


# ------------------------------------------------------------------ 5 / 6 / 7 losses


def test_hard_soft_exposure_losses_independent():
    nll = torch.tensor([1.0, 3.0, 2.0, 4.0])
    gold = torch.tensor([False, False, False, True])
    weight = torch.tensor([1 / 3, 1 / 3, 1 / 3, 0.0])
    hard = hard_sid_loss(nll, gold)
    soft = soft_sid_loss(nll, weight, 1)
    assert float(hard) == pytest.approx(4.0)
    assert float(soft) == pytest.approx(2.0)

    first = torch.zeros(2, 10)
    first[0, 2] = 10.0
    first[1, 3] = 10.0
    ids = torch.tensor([2, 3, 4])
    q_full = level1_student_probs(first, ids, probability_support="full_vocab")
    q_sid = level1_student_probs(first, ids, probability_support="valid_sid")
    assert q_full.shape == (2, 3)
    # The two supports must be allowed to differ (do not silently swap).
    p = torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    exp = exposure_alignment_loss(q_sid, p)
    assert torch.isfinite(exp) and exp >= 0
    total = distill_total_loss(hard, soft, exp, hard_weight=0.5, exposure_weight=0.1)
    assert torch.isfinite(total)


def test_hard_weight_one_is_plain_sid_sft():
    nll = torch.tensor([2.5, 1.0, 4.0])
    gold = torch.tensor([False, False, True])
    weight = torch.tensor([0.5, 0.5, 0.0])
    hard = hard_sid_loss(nll, gold)
    soft = soft_sid_loss(nll, weight, 1)
    exposure = torch.tensor(7.0)
    loss = distill_total_loss(hard, soft, exposure, hard_weight=1.0, exposure_weight=0.0)
    assert float(loss) == pytest.approx(float(hard))
    assert float(loss) == pytest.approx(4.0)


def test_full_vocab_vs_valid_sid_are_distinct():
    logits = torch.zeros(1, 8)
    logits[0, 0] = 4.0  # non-SID mass
    logits[0, 3] = 1.0
    logits[0, 4] = 1.0
    ids = torch.tensor([3, 4])
    q_full = level1_student_probs(logits, ids, probability_support="full_vocab")
    q_sid = level1_student_probs(logits, ids, probability_support="valid_sid")
    assert not torch.allclose(q_full, q_sid)
    assert torch.allclose(q_sid.sum(-1), torch.ones(1))
    assert float(q_full.sum()) < 1.0


# ------------------------------------------------------------------ 8 / 9 train vs val


def test_transition_train_only_selects_on_val(tmp_path):
    mapping = {k: [i, 0] for i, k in enumerate("abcd")}
    table = SidTable(_write_sid_table(tmp_path / "tv", mapping))
    train = [_prompt_row(["a"], "b"), _prompt_row(["b"], "c")]
    val = [_prompt_row(["c"], "d")]
    cfg = {
        "train": {
            "transition": {
                "history_max_length": 4,
                "embedding_dim": 8,
                "hidden_dim": 16,
                "epochs": 2,
                "batch_size": 2,
                "learning_rate": 0.05,
                "dropout": 0.0,
            }
        },
        "sid": {},
        "seed": 0,
    }
    summary = run_transition(
        cfg=cfg,
        sid_table=table,
        catalog=_catalog(list(mapping)),
        train_examples=train,
        val_examples=val,
        output_dir=tmp_path / "tr",
        logger=_Log(),
    )
    assert summary["n_train"] == 2
    assert summary["n_val"] == 1
    blob = torch.load(summary["checkpoint"], map_location="cpu", weights_only=False)
    assert blob["data"]["split"]["train"] == "train_examples"
    assert blob["data"]["split"]["val"] == "val_examples"
    assert blob["sid_fingerprint"]["config_hash"] == table.fingerprint()["config_hash"]
    assert "best_val_joint_nll" in blob["metrics"]


# ------------------------------------------------------------------ 10 smoke


def test_transition_and_distill_two_step_smoke(tmp_path):
    from llm4rec.trainers.sid_distill import run_sid_distill

    mapping = {f"i{k}": [k % 3, k % 2] for k in range(6)}
    table = SidTable(_write_sid_table(tmp_path / "sm", mapping))
    items = list(mapping)
    train = [_prompt_row([items[i], items[i + 1]], items[i + 2]) for i in range(3)]
    val = [_prompt_row([items[0]], items[1])]
    cfg = {
        "seed": 0,
        "sid": {},
        "hardware": {},
        "optimization": {"generation": {}},
        "checkpoint": {"save_steps": None},
        "train": {
            "transition": {
                "history_max_length": 4,
                "embedding_dim": 8,
                "hidden_dim": 16,
                "epochs": 1,
                "max_steps": 2,
                "batch_size": 2,
                "dropout": 0.0,
                "learning_rate": 0.05,
            },
            "distill": {
                "hard_weight": 0.5,
                "samples_per_prompt": 2,
                "exposure_weight": 0.1,
                "probability_support": "full_vocab",
                "epochs": 1,
                "max_steps": 2,
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "global_batch_size": 1,
                "learning_rate": 1e-3,
                "logging_steps": 1,
                "eval_steps": None,
                "max_seq_length": 32,
            },
        },
    }
    tr = run_transition(
        cfg=cfg,
        sid_table=table,
        catalog=_catalog(items),
        train_examples=train,
        val_examples=val,
        output_dir=tmp_path / "transition",
        logger=_Log(),
    )
    assert Path(tr["checkpoint"]).is_file()

    tok = _TinyTok()
    # Reserve ids for every SID token the collator will emit.
    for item, codes in mapping.items():
        for layer, code in enumerate(codes):
            tok.convert_tokens_to_ids(sid_token(layer, code, table.prefixes))
    model = _TinyLM(vocab=64)
    out = run_sid_distill(
        cfg=cfg,
        model=model,
        tokenizer=tok,
        sid_table=table,
        catalog=_catalog(items),
        train_examples=train,
        eval_examples=val,
        output_dir=tmp_path / "distill",
        logger=_Log(),
        runtime=_RT(),
        artifacts={"transition_checkpoint": tr["checkpoint"]},
    )
    assert out["checkpoint"].endswith("final")
    assert (Path(out["checkpoint"]) / "pytorch_model.bin").is_file()
    assert out["metrics"]["train_steps"] == 2


# ------------------------------------------------------------------ 11 GRPO regression


def test_minionerec_grpo_stages_still_validate():
    cfg = validate(to_dict(compose("smoke_minionerec")))
    assert cfg["stages"] == ["sft", "eval", "rl", "eval"]
    assert "rl" in (cfg.get("train") or {})
    from llm4rec.pipeline import MiniOneRecPipeline, Pipeline

    assert hasattr(MiniOneRecPipeline, "run_rl")
    src = Path("src/llm4rec/pipeline.py").read_text(encoding="utf-8")
    assert 'elif stage == "rl":' in src
    assert 'elif stage == "transition":' in src
    assert 'elif stage == "distill":' in src
    # Base pipeline still rejects SDA stages on other routes.
    dummy = Pipeline({}, Path("."), _Log())
    dummy.route = "recr1"
    with pytest.raises(ConfigurationError, match="transition"):
        dummy.run_transition(SimpleNamespace())


def test_minionerec_sda_stages_validate():
    cfg = validate(to_dict(compose("minionerec_sda_qwen05b")))
    assert cfg["stages"] == ["sft", "eval", "transition", "distill", "eval"]
    assert "transition" in cfg["train"]
    assert cfg["train"]["distill"]["samples_per_prompt"] == 8
    assert cfg["train"]["distill"]["probability_support"] == "full_vocab"


def test_recr1_rejects_transition_stage():
    cfg = to_dict(compose("smoke_recr1"))
    cfg["stages"] = ["sft", "transition", "eval"]
    with pytest.raises(ConfigurationError, match="transition"):
        validate(cfg)


# ------------------------------------------------------------------ 12 exposure single vs multi-rank definition


def test_exposure_mean_matches_sharded_definition():
    torch.manual_seed(2)
    q = F.softmax(torch.randn(8, 5), dim=-1)
    p = F.softmax(torch.randn(8, 5), dim=-1)
    full = exposure_alignment_loss(q, p)

    q_bar = q.mean(0).clamp(min=1e-9)
    p_bar = p.mean(0).clamp(min=1e-9)
    q_bar = q_bar / q_bar.sum()
    p_bar = p_bar / p_bar.sum()
    expected = (q_bar * (q_bar.log() - p_bar.log())).sum()
    assert torch.allclose(full, expected, atol=1e-6)

    # Two "ranks" of 4: size-weighted mean of local sums == global mean.
    q_sum = q[:4].sum(0) + q[4:].sum(0)
    p_sum = p[:4].sum(0) + p[4:].sum(0)
    q_bar2 = (q_sum / 8).clamp(min=1e-9)
    p_bar2 = (p_sum / 8).clamp(min=1e-9)
    q_bar2 = q_bar2 / q_bar2.sum()
    p_bar2 = p_bar2 / p_bar2.sum()
    sharded = (q_bar2 * (q_bar2.log() - p_bar2.log())).sum()
    assert torch.allclose(full, sharded, atol=1e-6)
