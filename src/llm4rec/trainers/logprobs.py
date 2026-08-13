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
    logits = model(full).logits[0]
    start = prompt_ids.shape[0] - 1
    end = full.shape[1] - 1
    target_logits = logits[start:end]
    log_probs = F.log_softmax(target_logits.float(), dim=-1)
    return log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)


def _score_padded_batch(
    model: Any,
    prompts: Sequence[torch.Tensor],
    completions: Sequence[torch.Tensor],
    *,
    pad_token_id: int | None = None,
    pad_to_multiple_of: int | None = None,
) -> list[torch.Tensor]:
    """Score N (prompt, completion) pairs in one forward."""
    assert len(prompts) == len(completions)
    if not completions:
        return []
    if len(completions) == 1:
        return [sequence_logprobs(model, prompts[0], completions[0])]

    device = prompts[0].device
    pad_id = 0 if pad_token_id is None else int(pad_token_id)
    prompt_lens = [int(p.numel()) for p in prompts]
    comp_lens = [int(c.numel()) for c in completions]
    max_seq = max(pl + cl for pl, cl in zip(prompt_lens, comp_lens, strict=True))
    max_seq = _pad_len(max_seq, pad_to_multiple_of)

    batch = len(completions)
    input_ids = torch.full((batch, max_seq), pad_id, dtype=prompts[0].dtype, device=device)
    attention_mask = torch.zeros((batch, max_seq), dtype=torch.long, device=device)

    for i, (prompt, comp) in enumerate(zip(prompts, completions, strict=True)):
        pl, cl = prompt_lens[i], comp_lens[i]
        input_ids[i, :pl] = prompt
        attention_mask[i, :pl] = 1
        if cl > 0:
            input_ids[i, pl : pl + cl] = comp
            attention_mask[i, pl : pl + cl] = 1

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = F.log_softmax(out.logits.float(), dim=-1)

    results: list[torch.Tensor] = []
    for i, (pl, cl) in enumerate(zip(prompt_lens, comp_lens, strict=True)):
        if cl == 0:
            results.append(completions[i].new_empty((0,), dtype=torch.float32))
            continue
        positions = torch.arange(pl - 1, pl - 1 + cl, device=device)
        results.append(log_probs[i, positions, completions[i]])
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
