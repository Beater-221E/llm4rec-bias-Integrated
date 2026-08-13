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

    Upstream ``GenerationConfig`` uses ``do_sample=True``, ``temperature=1.0``,
    ``num_beams=num_generations``, ``num_return_sequences=num_generations``.
    Reproduction must not force deterministic ``do_sample=False``.
    """

    def __init__(
        self,
        sid_table: Any,
        *,
        max_new_tokens: int | None = None,
        do_sample: bool = True,
        temperature: float = 1.0,
        length_penalty: float = 0.0,
        beam_search: bool = True,
    ) -> None:
        self.table = sid_table
        self.max_new_tokens = int(max_new_tokens or sid_table.levels + 2)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.length_penalty = float(length_penalty)
        self.beam_search = bool(beam_search)

    @torch.no_grad()
    def __call__(
        self, model: Any, tokenizer: Any, example: dict[str, Any], group_size: int
    ) -> Rollout:
        device = next(model.parameters()).device
        prompt_ids = _encode(tokenizer, example["prompt"]).to(device)
        eos = tokenizer.eos_token_id
        prompt_len = prompt_ids.shape[0]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "num_return_sequences": group_size,
            "prefix_allowed_tokens_fn": self.table.prefix_allowed_fn(
                tokenizer, prompt_len, eos
            ),
            "pad_token_id": eos,
        }
        if self.beam_search:
            gen_kwargs.update(
                {
                    "num_beams": group_size,
                    "do_sample": self.do_sample,
                    "temperature": self.temperature if self.do_sample else None,
                    "length_penalty": self.length_penalty,
                    "early_stopping": True,
                }
            )
            if not self.do_sample:
                gen_kwargs.pop("temperature", None)
        else:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                }
            )

        output = model.generate(prompt_ids.unsqueeze(0), **gen_kwargs)

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
        use_cache: bool = True,
        cache_implementation: str | None = None,
        kv_choice: Any = None,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)
        self.use_cache = bool(use_cache)
        self.cache_implementation = cache_implementation
        self.kv_choice = kv_choice
        self.cfg = cfg

    @torch.no_grad()
    def __call__(
        self, model: Any, tokenizer: Any, example: dict[str, Any], group_size: int
    ) -> Rollout:
        device = next(model.parameters()).device
        prompt_ids = _encode(tokenizer, example["prompt"]).to(device)
        eos = tokenizer.eos_token_id
        prompt_len = prompt_ids.shape[0]
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "num_return_sequences": group_size,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "pad_token_id": eos,
            "use_cache": self.use_cache,
        }
        if self.cache_implementation:
            gen_kwargs["cache_implementation"] = self.cache_implementation
        try:
            output = model.generate(prompt_ids.unsqueeze(0), **gen_kwargs)
        except Exception as exc:
            # Fall back to dynamic if static cache unsupported
            gen_kwargs.pop("cache_implementation", None)
            gen_kwargs["use_cache"] = True
            if self.kv_choice is not None and hasattr(self.kv_choice, "fallback_to_dynamic"):
                self.kv_choice.fallback_to_dynamic(str(exc))
                self.cache_implementation = None
                if self.cfg is not None:
                    from llm4rec.runtime.kv_cache import persist_kv_choice

                    persist_kv_choice(self.cfg, self.kv_choice)
            output = model.generate(prompt_ids.unsqueeze(0), **gen_kwargs)
        completions, texts = [], []
        for sequence in output:
            comp = sequence[prompt_len:]
            if eos is not None:
                nonpad = (comp != eos).nonzero()
                comp = comp[: int(nonpad[-1]) + 1] if nonpad.numel() else comp[:0]
            completions.append(comp)
            texts.append(tokenizer.decode(comp, skip_special_tokens=True))
        return Rollout(
            prompt_ids=prompt_ids,
            completion_ids=completions,
            texts=texts,
            example=example,
        )
