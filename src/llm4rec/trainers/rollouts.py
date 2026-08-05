"""各路线的采样策略。GRPO 训练循环通过这里注入路线差异。"""

from __future__ import annotations

from typing import Any

import torch

from llm4rec.trainers.grpo import Rollout


def _encode(tokenizer: Any, prompt: Any) -> torch.Tensor:
    if isinstance(prompt, str):
        return tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    encoded = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, return_tensors="pt"
    )
    ids = encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
    return ids[0]


class ConstrainedBeamRollout:
    """MiniOneRec：约束 beam 采样（官方 ``--beam_search True``）。

    每条 beam 都走全库 SID 前缀树，所以采样出来的 G 条**全是合法且互不相同**
    的 SID。这一点对 reward 很关键：官方的 ndcg_rule_reward 是按组内次序给
    位次奖励的，前提就是这一组本身是一个有序的候选列表。
    """

    def __init__(self, sid_table: Any, *, max_new_tokens: int | None = None) -> None:
        self.table = sid_table
        self.max_new_tokens = int(max_new_tokens or sid_table.levels + 2)

    @torch.no_grad()
    def __call__(
        self, model: Any, tokenizer: Any, example: dict[str, Any], group_size: int
    ) -> Rollout:
        device = next(model.parameters()).device
        prompt_ids = _encode(tokenizer, example["prompt"]).to(device)
        eos = tokenizer.eos_token_id
        prompt_len = prompt_ids.shape[0]

        output = model.generate(
            prompt_ids.unsqueeze(0),
            max_new_tokens=self.max_new_tokens,
            num_beams=group_size,
            num_return_sequences=group_size,
            prefix_allowed_tokens_fn=self.table.prefix_allowed_fn(
                tokenizer, prompt_len, eos
            ),
            do_sample=False,
            early_stopping=True,
            pad_token_id=eos,
        )

        completions, texts = [], []
        for sequence in output:
            comp = sequence[prompt_len:]
            comp = comp[comp != eos] if eos is not None else comp
            completions.append(comp)
            texts.append(tokenizer.decode(sequence[prompt_len:], skip_special_tokens=False))

        return Rollout(
            prompt_ids=prompt_ids,
            completion_ids=completions,
            texts=texts,
            example=example,
        )


class SamplingRollout:
    """Rec-R1：普通温度采样（官方 temperature=0.6 / top_p=0.95 / n=12）。"""

    def __init__(
        self,
        *,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
    ) -> None:
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)

    @torch.no_grad()
    def __call__(
        self, model: Any, tokenizer: Any, example: dict[str, Any], group_size: int
    ) -> Rollout:
        device = next(model.parameters()).device
        prompt_ids = _encode(tokenizer, example["prompt"]).to(device)
        prompt_len = prompt_ids.shape[0]
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

        output = model.generate(
            prompt_ids.unsqueeze(0),
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=group_size,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=pad,
        )

        completions, texts = [], []
        for sequence in output:
            comp = sequence[prompt_len:]
            if pad is not None:
                # 去掉尾部 padding，但保留中间可能出现的同 id token
                nonpad = (comp != pad).nonzero()
                if nonpad.numel():
                    comp = comp[: int(nonpad[-1]) + 1]
                else:
                    comp = comp[:0]
            completions.append(comp)
            texts.append(tokenizer.decode(comp, skip_special_tokens=True))

        return Rollout(
            prompt_ids=prompt_ids,
            completion_ids=completions,
            texts=texts,
            example=example,
        )
