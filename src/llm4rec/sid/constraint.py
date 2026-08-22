"""SID constrained decoding without per-beam ``prefix_allowed_tokens_fn``.

HF calls ``prefix_allowed_tokens_fn(batch_id, ids)`` once per beam per step and
``.tolist()`` syncs GPU→CPU each time. A ``LogitsProcessor`` runs once per step
for the whole beam batch and only copies the few newly generated SID tokens.
"""

from __future__ import annotations

from typing import Any

import torch


class SidPrefixLogitsProcessor:
    """Mask logits so every beam stays on the SID prefix trie."""

    def __init__(self, table: Any, tokenizer: Any, prompt_len: int, eos_id: int) -> None:
        self.root, self.allowed = table.cached_trie(tokenizer, eos_id)
        self.prompt_len = int(prompt_len)
        self.eos_id = int(eos_id)

    def bind(self, prompt_len: int) -> SidPrefixLogitsProcessor:
        self.prompt_len = int(prompt_len)
        return self

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        gen = input_ids[:, self.prompt_len :]
        paths = (
            gen.detach().to(device="cpu", non_blocking=False).tolist()
            if gen.numel()
            else [[] for _ in range(int(input_ids.shape[0]))]
        )
        neg = torch.finfo(scores.dtype).min
        mask = torch.zeros(scores.shape, dtype=torch.bool, device=scores.device)
        eos = self.eos_id
        root = self.root
        allowed_map = self.allowed
        for row, path in enumerate(paths):
            node = root
            ok = True
            for token_id in path:
                node = node.get(int(token_id))
                if node is None:
                    ok = False
                    break
            ids = allowed_map.get(id(node)) if ok else None
            if not ids:
                mask[row, eos] = True
            else:
                mask[row, ids] = True
        return scores.masked_fill(~mask, neg)


def reset_generate_limits(model: Any, prompt_len: int, max_new_tokens: int, eos_id: int) -> None:
    """Use ``max_new_tokens`` only; drop a stale ``max_length``.

    HuggingFace warns on every ``generate()`` when both are set (this repo's
    eval loop calls generate once per example). A leftover small ``max_length``
    from warmup can also clip decode. Clearing it lets HF compute
    ``max_length = max_new_tokens + prompt_len`` internally.
    """
    del prompt_len
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return
    gc.max_new_tokens = int(max_new_tokens)
    try:
        gc.max_length = None
    except (TypeError, ValueError):
        if hasattr(gc, "__dict__"):
            gc.__dict__["max_length"] = None
    gc.eos_token_id = int(eos_id)
    gc.pad_token_id = int(eos_id)


def constraint_generate_kwargs(
    table: Any,
    tokenizer: Any,
    prompt_len: int,
    eos_id: int,
) -> dict[str, Any]:
    from transformers import LogitsProcessorList

    processor = SidPrefixLogitsProcessor(table, tokenizer, prompt_len, eos_id)
    return {
        "logits_processor": LogitsProcessorList([processor]),
        "pad_token_id": eos_id,
    }
