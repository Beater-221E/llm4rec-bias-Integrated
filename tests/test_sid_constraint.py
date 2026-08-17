"""SID prefix trie cache + logits-processor vs prefix_allowed_fn."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from llm4rec.sid.constraint import SidPrefixLogitsProcessor
from llm4rec.sid.table import SidTable, sid_token


class _Tok:
    def __init__(self) -> None:
        self.ids: dict[str, int] = {}
        n = 10
        for prefix in "abc":
            for code in range(8):
                self.ids[sid_token("abc".index(prefix), code, ("a", "b", "c"))] = n
                n += 1
        self.eos_token_id = 2

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.ids[token]


def _write_sid_table(path: Path, mapping: dict[str, list[int]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    levels = len(next(iter(mapping.values())))
    codebook = max(max(c) for c in mapping.values()) + 1
    sid_map = {
        item: {
            "codes": codes,
            "sid": "".join(f"<{p}_{c}>" for p, c in zip("abc", codes)),
        }
        for item, codes in mapping.items()
    }
    (path / "item2sid.json").write_text(json.dumps(sid_map), encoding="utf-8")
    (path / "manifest.json").write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )


def _table(tmp_path: Path) -> SidTable:
    _write_sid_table(
        tmp_path,
        {
            "1": [0, 1, 2],
            "2": [0, 1, 3],
            "3": [1, 0, 0],
        },
    )
    return SidTable(tmp_path)


def test_cached_trie_reused(tmp_path):
    table = _table(tmp_path)
    tok = _Tok()
    first = table.cached_trie(tok, tok.eos_token_id)
    second = table.cached_trie(tok, tok.eos_token_id)
    assert first[0] is second[0]
    assert first[1] is second[1]


def test_logits_processor_matches_prefix_allowed(tmp_path):
    table = _table(tmp_path)
    tok = _Tok()
    eos = tok.eos_token_id
    prompt = [9, 9, 9]
    prompt_len = len(prompt)
    fn = table.prefix_allowed_fn(tok, prompt_len, eos)
    proc = SidPrefixLogitsProcessor(table, tok, prompt_len, eos)
    a0 = tok.convert_tokens_to_ids("<a_0>")
    b1 = tok.convert_tokens_to_ids("<b_1>")
    c2 = tok.convert_tokens_to_ids("<c_2>")
    prefixes = [
        [],
        [a0],
        [a0, b1],
        [a0, b1, c2],
        [999],
    ]
    vocab = 40
    for path in prefixes:
        ids = torch.tensor([prompt + path], dtype=torch.long)
        allowed = set(fn(0, ids[0]))
        scores = torch.arange(vocab, dtype=torch.float32).unsqueeze(0)
        out = proc(ids, scores.clone())
        kept = {i for i, v in enumerate(out[0].tolist()) if v > -1e4}
        assert kept == allowed


def test_reset_generate_limits_caps_stale_max_length():
    from llm4rec.sid.constraint import reset_generate_limits

    class _GC:
        max_length = 9
        max_new_tokens = 1
        eos_token_id = None
        pad_token_id = None

    class _M:
        generation_config = _GC()

    reset_generate_limits(_M(), prompt_len=400, max_new_tokens=5, eos_id=2)
    assert _M.generation_config.max_new_tokens == 5
    assert _M.generation_config.max_length >= 405
    assert _M.generation_config.eos_token_id == 2


def test_constraint_processor_bind_updates_prompt_len(tmp_path):
    table = _table(tmp_path)
    tok = _Tok()
    proc = table.constraint_processor(tok, 4, tok.eos_token_id)
    assert proc.prompt_len == 4
    assert proc.bind(12) is proc
    assert proc.prompt_len == 12
