"""MiniOneRec 路线的 Decoder：SID token → 约束 beam 解码 → ranked item list。

对齐官方 ``evaluate.py`` + ``LogitProcessor.py``：用全库 SID 的前缀树约束
每一步的候选 token，保证每条 beam 都是合法且唯一的 SID，非法率恒为 0。

输出统一成 ``RankedResult``，之后 bias 指标的计算就和另外两条路线完全共用。
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch

from llm4rec.decoders.base import Decoder
from llm4rec.eval.bias import RankedResult
from llm4rec.sid.table import SidTable
from llm4rec.tracking.progress import overwrite_progress

_LOG = logging.getLogger(__name__)


class ConstrainedBeamDecoder(Decoder):
    """约束 beam search：一次 generate 出 top-K 个互不相同的合法物品。"""

    name = "constrained_beam"

    def __init__(
        self,
        table: SidTable,
        *,
        num_beams: int = 20,
        max_new_tokens: int | None = None,
        fail_on_invalid: bool = True,
    ) -> None:
        self.table = table
        self.num_beams = int(num_beams)
        # 层数 + eos 的余量；约束解码下不会超
        self.max_new_tokens = int(max_new_tokens or table.levels + 2)
        self.fail_on_invalid = bool(fail_on_invalid)

    @torch.no_grad()
    def decode_batch(
        self,
        model: Any,
        tokenizer: Any,
        examples: Sequence[dict[str, Any]],
        *,
        top_k: int,
        progress_total: int | None = None,
        progress_dir: Any = None,
        progress_name: str | None = None,
    ) -> list[RankedResult]:
        from llm4rec.core import distributed as dist_utils

        device = next(model.parameters()).device
        eos_id = tokenizer.eos_token_id
        beams = max(int(top_k), self.num_beams)

        results: list[RankedResult] = []
        n_total = len(examples)
        ambiguity_before = int(getattr(self.table, "_ambiguity_count", 0) or 0)
        from transformers import LogitsProcessorList

        from llm4rec.sid.constraint import SidPrefixLogitsProcessor

        constraint = SidPrefixLogitsProcessor(self.table, tokenizer, 0, eos_id)
        with overwrite_progress(
            n_total,
            "eval",
            global_total=progress_total,
            progress_dir=progress_dir,
            name=progress_name or "eval/constrained_beam",
        ) as progress:
            for example in examples:
                input_ids = _encode_prompt(tokenizer, example["prompt"]).to(device)
                prompt_len = input_ids.shape[1]

                from llm4rec.sid.constraint import reset_generate_limits

                reset_generate_limits(model, prompt_len, self.max_new_tokens, eos_id)
                output = model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=beams,
                    num_return_sequences=beams,
                    do_sample=False,
                    early_stopping=False,
                    length_penalty=0.0,
                    repetition_penalty=1.0,
                    logits_processor=LogitsProcessorList([constraint.bind(prompt_len)]),
                    eos_token_id=eos_id,
                    pad_token_id=eos_id,
                )

                ranked: list[str] = []
                n_invalid = 0
                for sequence in output:
                    text = tokenizer.decode(
                        sequence[prompt_len:], skip_special_tokens=False
                    )
                    item = self.table.parse(text)
                    if item is None:
                        n_invalid += 1
                        continue
                    if item not in ranked:
                        ranked.append(item)
                    if len(ranked) >= top_k:
                        break

                if n_invalid and self.fail_on_invalid:
                    # 约束解码下不该出现非法 SID；出现了说明前缀树和词表没对上，
                    # 这时候静默降级会让 bias 数字悄悄失真，所以直接炸。
                    raise RuntimeError(
                        f"约束 beam 解码产生了 {n_invalid} 条非法 SID —— "
                        f"通常是 tokenizer 没加 SID token 或 embedding 没 resize。"
                        f"（要临时放行请设 decoder.fail_on_invalid=false）"
                    )

                results.append(
                    RankedResult(
                        user_id=str(example.get("user_id") or ""),
                        ranked_items=ranked,
                        target_item=str(example["target_item"]),
                        history=[str(i) for i in (example.get("history") or [])],
                        valid=n_invalid == 0 and bool(ranked),
                    )
                )
                progress.update(1)

        ambiguity_delta = int(getattr(self.table, "_ambiguity_count", 0) or 0) - ambiguity_before
        if dist_utils.is_main() and ambiguity_delta > 0:
            _LOG.debug(
                "[constrained_beam] SID ambiguity resolutions this shard: %d",
                ambiguity_delta,
            )
        return results


def _encode_prompt(tokenizer: Any, prompt: Any) -> torch.Tensor:
    """Match SFT: ``apply_chat_template(..., tokenize=True)`` when possible."""
    if isinstance(prompt, str):
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        ids = encoded["input_ids"] if not isinstance(encoded, torch.Tensor) else encoded
    else:
        ids = None
        try:
            encoded = tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, tokenize=True, return_tensors="pt"
            )
            ids = encoded["input_ids"] if hasattr(encoded, "input_ids") else encoded
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
        except Exception:
            ids = None
        if ids is None or (isinstance(ids, torch.Tensor) and ids.numel() == 0):
            try:
                text = tokenizer.apply_chat_template(
                    prompt, add_generation_prompt=True, tokenize=False
                )
            except Exception:
                text = ""
            if not str(text).strip():
                text = "\n".join(
                    str(m.get("content") or "")
                    for m in prompt
                    if isinstance(m, dict)
                )
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
            ids = encoded["input_ids"] if not isinstance(encoded, torch.Tensor) else encoded
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    if ids.numel() == 0 or ids.shape[-1] == 0:
        raise RuntimeError("eval prompt encoded to empty input_ids")
    return ids
