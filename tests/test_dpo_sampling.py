"""DPO batch sampling: ``sample_reasonings_many`` vs single-sample path."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from llm4rec.trainers.dpo import sample_reasonings, sample_reasonings_many


class TinyCausal(nn.Module):
    def __init__(self, vocab: int = 64, dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.out = nn.Linear(dim, vocab)
        self.config = type("C", (), {"use_cache": True})()

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=4, num_return_sequences=1, **kw):
        b = input_ids.shape[0]
        outs = []
        for _ in range(num_return_sequences):
            extra = torch.randint(
                0, self.embed.num_embeddings, (b, max_new_tokens), device=input_ids.device
            )
            outs.append(torch.cat([input_ids, extra], dim=1))
        return torch.cat(outs, dim=0)


class TinyTok:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, prompt, add_generation_prompt=True, return_tensors=None):
        ids = torch.tensor([[2, 3, 4, 5]])
        return ids if return_tensors == "pt" else ids[0].tolist()

    def decode(self, ids, skip_special_tokens=True):
        return "tok"


def _examples(n: int) -> list[dict[str, object]]:
    return [
        {"user_id": i, "prompt": [{"role": "user", "content": f"hi {i}"}]}
        for i in range(n)
    ]


def test_batch_matches_single_for_one_example():
    model = TinyCausal()
    tok = TinyTok()
    ex = _examples(1)
    many = sample_reasonings_many(
        model, tok, ex, n=3, temperature=1.0, max_new_tokens=4
    )
    single = sample_reasonings(
        model, tok, ex[0], n=3, temperature=1.0, max_new_tokens=4
    )
    assert len(many) == 1
    prompt_ids, completions, texts = many[0]
    assert torch.equal(prompt_ids, single[0])
    assert len(completions) == 3
    assert len(texts) == 3
    assert len(single[1]) == 3


def test_batch_shapes_for_multiple_examples():
    model = TinyCausal()
    tok = TinyTok()
    examples = _examples(4)
    results = sample_reasonings_many(
        model, tok, examples, n=2, temperature=1.0, max_new_tokens=4
    )
    assert len(results) == 4
    for (prompt_ids, completions, texts), ex in zip(results, examples, strict=True):
        assert int(prompt_ids.shape[0]) == 4  # TinyTok prompt length
        assert len(completions) == 2
        assert len(texts) == 2
        assert ex["user_id"] in (0, 1, 2, 3)


def test_batch_chunks_on_oom_fallback_path():
    model = TinyCausal()
    tok = TinyTok()
    examples = _examples(3)
    results = sample_reasonings_many(
        model, tok, examples, n=1, temperature=1.0, max_new_tokens=4, max_batch=1
    )
    assert len(results) == 3


def test_empty_input_returns_empty():
    model = TinyCausal()
    tok = TinyTok()
    assert sample_reasonings_many(model, tok, [], n=2, temperature=1.0, max_new_tokens=4) == []


def test_batch_split_reuses_same_generation_shape():
    """Batched output must split as output[i*n:(i+1)*n] — verify via counts."""
    model = TinyCausal()
    tok = TinyTok()
    n = 3
    examples = _examples(5)
    results = sample_reasonings_many(
        model, tok, examples, n=n, temperature=1.0, max_new_tokens=4
    )
    assert all(len(completions) == n for _, completions, _ in results)
    # Completions are truncated at the first pad/eos; never exceed prompt_len+max_new_tokens
    assert all(int(c.shape[0]) <= 8 for _, completions, _ in results for c in completions)
