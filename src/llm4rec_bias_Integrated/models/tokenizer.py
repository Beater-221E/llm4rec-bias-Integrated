"""Tokenizer helpers."""

from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_tokenizer(
    checkpoint: str,
    *,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(
        checkpoint,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
