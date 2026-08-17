"""Batched sequence log-probabilities for GRPO / DPO.

Supports:
  - single-prompt G completions
  - multi-prompt B×G completions (heterogeneous prompts)
  - preference minibatch of 2B sequences (chosen/rejected)
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F


def _pad_len(n: int, multiple: int | None) -> int:
    if not multiple or multiple <= 1:
        return n
    rem = n % multiple
    return n if rem == 0 else n + (multiple - rem)


def sequence_logprobs(
    model: Any,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-sequence API (kept for parity tests / tiny paths)."""
    full = torch.cat([prompt_ids, completion_ids]).unsqueeze(0)
    cl = int(completion_ids.numel())
    keep = cl + 1
    try:
        logits = model(input_ids=full, logits_to_keep=keep).logits
    except TypeError:
        logits = model(input_ids=full).logits
    start = 0 if int(logits.shape[1]) == keep else prompt_ids.shape[0] - 1
    token_logits = logits[0, start : start + cl, :].float()
    log_probs = F.log_softmax(token_logits, dim=-1)
    return log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)


def _infer_vocab_size(model: Any) -> int:
    core = getattr(model, "module", model)
    getter = getattr(core, "get_output_embeddings", None)
    if callable(getter):
        emb = getter()
        if emb is not None and hasattr(emb, "weight"):
            return int(emb.weight.shape[0])
    cfg = getattr(core, "config", None)
    vocab = getattr(cfg, "vocab_size", None) if cfg is not None else None
    return int(vocab) if vocab else 0


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def vram_logprob_budget_bytes(floor: int = 512 * 1024 * 1024) -> int:
    """Cap fp32 logits; keep most free VRAM for activations + autograd."""
    if not torch.cuda.is_available():
        return int(floor)
    try:
        free, _ = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001
        return int(floor)
    return max(int(floor), int(free * 0.12))


def logprob_chunk_size(
    batch: int,
    seq_len: int,
    vocab: int,
    *,
    budget_bytes: int = 512 * 1024 * 1024,
) -> int:
    """Keep ``[chunk, seq, vocab]`` fp32 logits under ``budget_bytes``."""
    if batch <= 1 or vocab <= 0 or seq_len <= 0:
        return max(1, batch)
    per_row = max(1, int(seq_len) * int(vocab) * 4)
    return max(1, min(int(batch), int(budget_bytes) // per_row))


def _score_padded_batch(
    model: Any,
    prompts: Sequence[torch.Tensor],
    completions: Sequence[torch.Tensor],
    *,
    pad_token_id: int | None = None,
    pad_to_multiple_of: int | None = None,
    max_chunk: int | None = None,
) -> list[torch.Tensor]:
    """Score N (prompt, completion) pairs in one forward."""
    assert len(prompts) == len(completions)
    if not completions:
        return []

    prompt_lens = [int(p.numel()) for p in prompts]
    comp_lens = [int(c.numel()) for c in completions]
    max_seq = max(pl + cl for pl, cl in zip(prompt_lens, comp_lens, strict=True))
    max_seq = _pad_len(max_seq, pad_to_multiple_of)
    batch = len(completions)
    chunk = int(max_chunk) if max_chunk is not None else logprob_chunk_size(
        batch,
        max_seq,
        _infer_vocab_size(model),
        budget_bytes=vram_logprob_budget_bytes(),
    )
    if batch > chunk:
        out: list[torch.Tensor] = []
        for start in range(0, batch, chunk):
            end = min(batch, start + chunk)
            out.extend(
                _score_padded_batch(
                    model,
                    prompts[start:end],
                    completions[start:end],
                    pad_token_id=pad_token_id,
                    pad_to_multiple_of=pad_to_multiple_of,
                    max_chunk=chunk,
                )
            )
        return out

    try:
        return _forward_padded_batch(
            model,
            prompts,
            completions,
            prompt_lens=prompt_lens,
            comp_lens=comp_lens,
            max_seq=max_seq,
            pad_token_id=pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if not _is_cuda_oom(exc) or batch <= 1:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mid = max(1, batch // 2)
        return _score_padded_batch(
            model,
            prompts[:mid],
            completions[:mid],
            pad_token_id=pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
            max_chunk=mid,
        ) + _score_padded_batch(
            model,
            prompts[mid:],
            completions[mid:],
            pad_token_id=pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
            max_chunk=mid,
        )


def _forward_padded_batch(
    model: Any,
    prompts: Sequence[torch.Tensor],
    completions: Sequence[torch.Tensor],
    *,
    prompt_lens: list[int],
    comp_lens: list[int],
    max_seq: int,
    pad_token_id: int | None,
    pad_to_multiple_of: int | None,
) -> list[torch.Tensor]:
    del pad_to_multiple_of
    batch = len(completions)
    device = prompts[0].device
    pad_id = 0 if pad_token_id is None else int(pad_token_id)
    input_ids = torch.full((batch, max_seq), pad_id, dtype=prompts[0].dtype, device=device)
    attention_mask = torch.zeros((batch, max_seq), dtype=torch.long, device=device)

    for i, (prompt, comp) in enumerate(zip(prompts, completions, strict=True)):
        pl, cl = prompt_lens[i], comp_lens[i]
        input_ids[i, :pl] = prompt
        attention_mask[i, :pl] = 1
        if cl > 0:
            input_ids[i, pl : pl + cl] = comp
            attention_mask[i, pl : pl + cl] = 1

    keep = max(1, max_seq - min(prompt_lens) + 1)
    try:
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=keep,
        ).logits
    except TypeError:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # Models that ignore logits_to_keep still return [B, T, V].
    offset = max_seq - int(logits.shape[1]) if int(logits.shape[1]) < max_seq else 0
    results: list[torch.Tensor] = []
    for i, (pl, cl) in enumerate(zip(prompt_lens, comp_lens, strict=True)):
        if cl == 0:
            results.append(completions[i].new_empty((0,), dtype=torch.float32))
            continue
        # Softmax only completion rows — full [B, T, V] log_softmax doubles VRAM
        # when the SID-expanded vocab is ~1.5e5.
        start = pl - 1 - offset
        token_logits = logits[i, start : start + cl, :].float()
        log_probs = F.log_softmax(token_logits, dim=-1)
        results.append(log_probs.gather(-1, completions[i].unsqueeze(-1)).squeeze(-1))
    return results


def batched_sequence_logprobs(
    model: Any,
    prompt_ids: torch.Tensor,
    completion_ids: Sequence[torch.Tensor],
    *,
    pad_token_id: int | None = None,
    pad_to_multiple_of: int | None = None,
) -> list[torch.Tensor]:
    """Score many completions that share the same prompt in ONE forward."""
    if not completion_ids:
        return []
    prompts = [prompt_ids] * len(completion_ids)
    return _score_padded_batch(
        model,
        prompts,
        completion_ids,
        pad_token_id=pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )


def batched_multi_prompt_logprobs(
    model: Any,
    prompt_ids_list: Sequence[torch.Tensor],
    completion_ids: Sequence[torch.Tensor],
    *,
    pad_token_id: int | None = None,
    pad_to_multiple_of: int | None = None,
) -> list[torch.Tensor]:
    """Score B×G (or any) heterogeneous prompt/completion pairs in one forward."""
    return _score_padded_batch(
        model,
        prompt_ids_list,
        completion_ids,
        pad_token_id=pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )


def batched_pair_logprobs(
    model: Any,
    prompt_ids: torch.Tensor,
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    *,
    pad_token_id: int | None = None,
    pad_to_multiple_of: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score chosen + rejected in one forward; return sum log-probs."""
    lps = batched_sequence_logprobs(
        model,
        prompt_ids,
        [chosen, rejected],
        pad_token_id=pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    return lps[0].sum(), lps[1].sum()


def score_preference_batch(
    model: Any,
    prompts: Sequence[torch.Tensor],
    chosens: Sequence[torch.Tensor],
    rejecteds: Sequence[torch.Tensor],
    *,
    pad_token_id: int | None = None,
    pad_to_multiple_of: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score B preference pairs as 2B sequences in one forward.

    Returns ``(policy_chosen_sums[B], policy_rejected_sums[B])``.
    """
    b = len(prompts)
    if b == 0:
        empty = torch.zeros(0, dtype=torch.float32)
        return empty, empty
    assert len(chosens) == b and len(rejecteds) == b
    all_prompts = list(prompts) + list(prompts)
    all_comps = list(chosens) + list(rejecteds)
    lps = _score_padded_batch(
        model,
        all_prompts,
        all_comps,
        pad_token_id=pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    sums = torch.stack([lp.sum() for lp in lps])
    return sums[:b], sums[b:]
