"""Load causal LM with SID tokens added (regular tokens, not special)."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm4rec.components.model._impl.base import require_cuda, resolve_precision
from llm4rec.workflows.minionerec.semantic_ids.table import SidTable


def prepare_sid_model(
    checkpoint: str,
    table: SidTable,
    *,
    dtype: str = "auto",
    local_rank: int = 0,
) -> tuple[Any, Any, list[int]]:
    """Return (tokenizer, model, new_token_ids).

    SID tokens are **regular** added tokens so TRL GRPO decode with
    ``skip_special_tokens=True`` does not strip them.
    """
    require_cuda()
    precision = resolve_precision(dtype)
    tok = AutoTokenizer.from_pretrained(checkpoint)
    added = tok.add_tokens(table.tokens(), special_tokens=False)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, dtype=precision.dtype
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=precision.dtype
        )
    if len(tok) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tok))
    new_ids = [tok.convert_tokens_to_ids(t) for t in table.tokens()]
    if added:
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            mean = emb[: len(tok) - added].mean(0)
            for i in new_ids:
                emb[i] = mean + 0.02 * torch.randn_like(mean)
    model = model.to(f"cuda:{local_rank}")
    return tok, model, new_ids
