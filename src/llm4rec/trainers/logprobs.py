"""Batched sequence log-probabilities for GRPO / DPO.

Supports:
  - single-prompt G completions
  - multi-prompt B×G completions (heterogeneous prompts)
  - preference minibatch of 2B sequences (chosen/rejected)
"""

from __future__ import annotations

import gc
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


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _pack_left_padded(
    prompts: Sequence[torch.Tensor],
    completions: Sequence[torch.Tensor],
    prompt_lens: list[int],
    comp_lens: list[int],
    max_seq: int,
    pad_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Left-pad so completions sit on the right; ``logits_to_keep`` stays tiny.

    Right-padding made ``keep ≈ max_seq - min(prompt)``, materializing
    ``[B, ~T, V]`` SID-expanded logits (~153k) and blowing 80GB during distill.
    """
    batch = len(completions)
    device = prompts[0].device
    pad_id = 0 if pad_token_id is None else int(pad_token_id)
    input_ids = torch.full((batch, max_seq), pad_id, dtype=prompts[0].dtype, device=device)
    attention_mask = torch.zeros((batch, max_seq), dtype=torch.long, device=device)
    for i, (prompt, comp) in enumerate(zip(prompts, completions, strict=True)):
        pl, cl = prompt_lens[i], comp_lens[i]
        start = max_seq - pl - cl
        if pl:
            input_ids[i, start : start + pl] = prompt
        if cl:
            input_ids[i, start + pl : start + pl + cl] = comp
        attention_mask[i, start:max_seq] = 1
    keep = max(1, (max(comp_lens) if comp_lens else 0) + 1)
    return input_ids, attention_mask, keep


def _completion_logit_start(cl: int, logits_len: int, max_seq: int) -> int:
    """Index in ``logits[i]`` of the row that predicts the first completion token."""
    offset = max_seq - logits_len if logits_len < max_seq else 0
    return max_seq - cl - 1 - offset


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
    keep_est = max(1, (max(comp_lens) if comp_lens else 0) + 1)
    chunk = int(max_chunk) if max_chunk is not None else logprob_chunk_size(
        batch,
        keep_est,
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
        _release_cuda()
        if torch.is_grad_enabled():
            raise
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
    input_ids, attention_mask, keep = _pack_left_padded(
        prompts, completions, prompt_lens, comp_lens, max_seq, pad_token_id
    )
    try:
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=keep,
        ).logits
    except TypeError:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits_len = int(logits.shape[1])
    results: list[torch.Tensor] = []
    for i, cl in enumerate(comp_lens):
        if cl == 0:
            results.append(completions[i].new_empty((0,), dtype=torch.float32))
            continue
        start = _completion_logit_start(cl, logits_len, max_seq)
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


def sid_sequence_nll(
    model: Any,
    prompts: Sequence[torch.Tensor],
    completions: Sequence[torch.Tensor],
    *,
    pad_token_id: int | None = None,
    pad_to_multiple_of: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced SID completion NLL + first-token logits.

    Mapped from reference ``sid_distill.sequence_nll``, implemented on top of
    the same padded multi-prompt scorer as GRPO/DPO:

    * NLL only over SID completion tokens
    * heterogeneous prompt lengths
    * ``logits_to_keep`` when the model supports it
    * first-token logits (the position that predicts SID level-1)

    Returns ``(nll[N], first_logits[N, V])``.
    """
    if not completions:
        empty = torch.zeros(0, dtype=torch.float32)
        return empty, empty.view(0, 0)
    nll_rows, first_logits = _score_padded_batch_with_first_logits(
        model,
        prompts,
        completions,
        pad_token_id=pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    nll = torch.stack(
        [
            (-lp).sum() if lp.numel() else lp.new_zeros(())
            for lp in nll_rows
        ]
    )
    return nll, torch.stack(first_logits)


def _score_padded_batch_with_first_logits(
    model: Any,
    prompts: Sequence[torch.Tensor],
    completions: Sequence[torch.Tensor],
    *,
    pad_token_id: int | None = None,
    pad_to_multiple_of: int | None = None,
    max_chunk: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Like ``_score_padded_batch`` but also keeps first-completion-token logits."""
    assert len(prompts) == len(completions)
    if not completions:
        return [], []
    prompt_lens = [int(p.numel()) for p in prompts]
    comp_lens = [int(c.numel()) for c in completions]
    max_seq = max(pl + cl for pl, cl in zip(prompt_lens, comp_lens, strict=True))
    max_seq = _pad_len(max_seq, pad_to_multiple_of)
    batch = len(completions)
    keep_est = max(1, (max(comp_lens) if comp_lens else 0) + 1)
    chunk = int(max_chunk) if max_chunk is not None else logprob_chunk_size(
        batch,
        keep_est,
        _infer_vocab_size(model),
        budget_bytes=vram_logprob_budget_bytes(),
    )
    if batch > chunk:
        nll_out: list[torch.Tensor] = []
        logit_out: list[torch.Tensor] = []
        for start in range(0, batch, chunk):
            end = min(batch, start + chunk)
            nll_part, logit_part = _score_padded_batch_with_first_logits(
                model,
                prompts[start:end],
                completions[start:end],
                pad_token_id=pad_token_id,
                pad_to_multiple_of=pad_to_multiple_of,
                max_chunk=chunk,
            )
            nll_out.extend(nll_part)
            logit_out.extend(logit_part)
        return nll_out, logit_out

    try:
        return _forward_padded_batch_with_first_logits(
            model,
            prompts,
            completions,
            prompt_lens=prompt_lens,
            comp_lens=comp_lens,
            max_seq=max_seq,
            pad_token_id=pad_token_id,
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if not _is_cuda_oom(exc) or batch <= 1:
            raise
        _release_cuda()
        if torch.is_grad_enabled():
            raise
        mid = max(1, batch // 2)
        left = _score_padded_batch_with_first_logits(
            model,
            prompts[:mid],
            completions[:mid],
            pad_token_id=pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
            max_chunk=mid,
        )
        right = _score_padded_batch_with_first_logits(
            model,
            prompts[mid:],
            completions[mid:],
            pad_token_id=pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
            max_chunk=mid,
        )
        return left[0] + right[0], left[1] + right[1]


def _forward_padded_batch_with_first_logits(
    model: Any,
    prompts: Sequence[torch.Tensor],
    completions: Sequence[torch.Tensor],
    *,
    prompt_lens: list[int],
    comp_lens: list[int],
    max_seq: int,
    pad_token_id: int | None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    input_ids, attention_mask, keep = _pack_left_padded(
        prompts, completions, prompt_lens, comp_lens, max_seq, pad_token_id
    )
    try:
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=keep,
        ).logits
    except TypeError:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits_len = int(logits.shape[1])
    token_lps: list[torch.Tensor] = []
    first_logits: list[torch.Tensor] = []
    vocab = int(logits.shape[-1])
    for i, cl in enumerate(comp_lens):
        if cl == 0:
            token_lps.append(completions[i].new_empty((0,), dtype=torch.float32))
            first_logits.append(logits.new_zeros((vocab,)))
            continue
        start = _completion_logit_start(cl, logits_len, max_seq)
        token_logits = logits[i, start : start + cl, :].float()
        log_probs = F.log_softmax(token_logits, dim=-1)
        token_lps.append(log_probs.gather(-1, completions[i].unsqueeze(-1)).squeeze(-1))
        first_logits.append(token_logits[0])
    return token_lps, first_logits


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
