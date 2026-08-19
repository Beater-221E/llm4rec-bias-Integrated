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
        self._constraint = None

    @torch.no_grad()
    def __call__(
        self, model: Any, tokenizer: Any, example: dict[str, Any], group_size: int
    ) -> Rollout:
        from transformers import LogitsProcessorList

        from llm4rec.sid.constraint import SidPrefixLogitsProcessor, reset_generate_limits

        device = next(model.parameters()).device
        prompt_ids = _encode(tokenizer, example["prompt"]).to(device)
        eos = tokenizer.eos_token_id
        prompt_len = prompt_ids.shape[0]
        if self._constraint is None:
            self._constraint = SidPrefixLogitsProcessor(self.table, tokenizer, prompt_len, eos)
        else:
            self._constraint.bind(prompt_len)

        reset_generate_limits(model, prompt_len, self.max_new_tokens, eos)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "num_return_sequences": group_size,
            "logits_processor": LogitsProcessorList([self._constraint]),
            "eos_token_id": eos,
            "pad_token_id": eos,
            "use_cache": True,
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
            # Official ReReTrainer: batch_decode(..., skip_special_tokens=True).
            # Keeping <|im_end|> makes rule_reward string-match miss every SID.
            texts.append(tokenizer.decode(comp, skip_special_tokens=True))

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
            output = self._generate_with_cache_fallback(
                model, prompt_ids.unsqueeze(0), None, gen_kwargs, exc
            )
        return self._pack_rollout(tokenizer, example, prompt_ids, output, prompt_len, eos)

    @torch.no_grad()
    def generate_many(
        self,
        model: Any,
        tokenizer: Any,
        examples: list[dict[str, Any]],
        group_size: int,
        *,
        max_batch: int | None = None,
    ) -> list[Rollout]:
        """Batched temperature sampling. Falls back to smaller chunks on OOM."""
        if not examples:
            return []
        if len(examples) == 1 or max_batch == 1:
            return [self(model, tokenizer, ex, group_size) for ex in examples]
        limit = int(max_batch) if max_batch and max_batch > 0 else len(examples)
        if len(examples) > limit:
            out: list[Rollout] = []
            for start in range(0, len(examples), limit):
                out.extend(
                    self.generate_many(
                        model,
                        tokenizer,
                        examples[start : start + limit],
                        group_size,
                        max_batch=limit,
                    )
                )
            return out
        try:
            return self._generate_padded_batch(model, tokenizer, examples, group_size)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(examples) <= 1:
                raise
            mid = max(1, len(examples) // 2)
            return self.generate_many(
                model, tokenizer, examples[:mid], group_size, max_batch=mid
            ) + self.generate_many(
                model, tokenizer, examples[mid:], group_size, max_batch=mid
            )

    def _generate_kwargs(self, tokenizer: Any, group_size: int) -> dict[str, Any]:
        eos = tokenizer.eos_token_id
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "num_return_sequences": group_size,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos,
            "eos_token_id": eos,
            "use_cache": self.use_cache,
        }
        if self.cache_implementation:
            kwargs["cache_implementation"] = self.cache_implementation
        return kwargs

    def _generate_with_cache_fallback(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        gen_kwargs: dict[str, Any],
        exc: Exception,
    ) -> torch.Tensor:
        gen_kwargs.pop("cache_implementation", None)
        gen_kwargs["use_cache"] = True
        if self.kv_choice is not None and hasattr(self.kv_choice, "fallback_to_dynamic"):
            self.kv_choice.fallback_to_dynamic(str(exc))
            self.cache_implementation = None
            if self.cfg is not None:
                from llm4rec.runtime.kv_cache import persist_kv_choice

                persist_kv_choice(self.cfg, self.kv_choice)
        if attention_mask is None:
            return model.generate(input_ids, **gen_kwargs)
        return model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

    def _generate_padded_batch(
        self,
        model: Any,
        tokenizer: Any,
        examples: list[dict[str, Any]],
        group_size: int,
    ) -> list[Rollout]:
        device = next(model.parameters()).device
        eos = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
        prompt_list = [_encode(tokenizer, ex["prompt"]).to(device) for ex in examples]
        prompt_lens = [int(p.numel()) for p in prompt_list]
        max_len = max(prompt_lens)
        batch = len(prompt_list)
        input_ids = torch.full((batch, max_len), int(pad_id), dtype=prompt_list[0].dtype, device=device)
        attention_mask = torch.zeros((batch, max_len), dtype=torch.long, device=device)
        for i, (prompt, n) in enumerate(zip(prompt_list, prompt_lens, strict=True)):
            input_ids[i, -n:] = prompt
            attention_mask[i, -n:] = 1
        gen_kwargs = self._generate_kwargs(tokenizer, group_size)
        try:
            output = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
        except Exception as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower():
                raise
            output = self._generate_with_cache_fallback(
                model, input_ids, attention_mask, gen_kwargs, exc
            )
        rollouts: list[Rollout] = []
        for i, example in enumerate(examples):
            seqs = output[i * group_size : (i + 1) * group_size]
            rollouts.append(
                self._pack_rollout(tokenizer, example, prompt_list[i], seqs, max_len, eos)
            )
        return rollouts

    def _pack_rollout(
        self,
        tokenizer: Any,
        example: dict[str, Any],
        prompt_ids: torch.Tensor,
        output: torch.Tensor,
        prompt_len: int,
        eos: int | None,
    ) -> Rollout:
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
