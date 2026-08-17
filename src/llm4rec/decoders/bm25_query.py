"""Rec-R1 路线的 Decoder：LLM 生成 query → BM25 检索 → ranked item list。"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from llm4rec.decoders.base import Decoder
from llm4rec.eval.bias import RankedResult
from llm4rec.tracking.progress import overwrite_progress
from llm4rec.trainers.rewards import extract_answer, parse_query, validate_structure


class BM25QueryDecoder(Decoder):
    """贪心解一条 query，交给检索器出 top-K。"""

    name = "bm25_query"

    def __init__(
        self,
        retriever: Any,
        *,
        answer_key: str = "query",
        max_new_tokens: int = 512,
        retrieval_top_k: int = 100,
    ) -> None:
        self.retriever = retriever
        self.answer_key = answer_key
        self.max_new_tokens = int(max_new_tokens)
        self.retrieval_top_k = int(retrieval_top_k)

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
        device = next(model.parameters()).device
        pad = tokenizer.pad_token_id or tokenizer.eos_token_id
        depth = max(int(top_k), self.retrieval_top_k)

        results: list[RankedResult] = []
        n_total = len(examples)
        with overwrite_progress(
            n_total,
            "eval",
            global_total=progress_total,
            progress_dir=progress_dir,
            name=progress_name or "eval/bm25_query",
        ) as progress:
            for example in examples:
                encoded = tokenizer.apply_chat_template(
                    example["prompt"], add_generation_prompt=True, return_tensors="pt"
                )
                ids = (
                    encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
                ).to(device)
                prompt_len = ids.shape[1]

                output = model.generate(
                    ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad,
                )
                text = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)

                query = parse_query(extract_answer(text), self.answer_key)
                valid = validate_structure(text) and query is not None

                # 格式非法 → 空列表，但仍然产出一条记录（valid=False）。
                # 跳过的话 valid_rate 和所有指标的分母都会算错。
                ranked = self.retriever.search(query, top_k=depth)[:top_k] if query else []

                results.append(
                    RankedResult(
                        user_id=str(example.get("user_id") or ""),
                        ranked_items=[str(i) for i in ranked],
                        target_item=str(example["target_item"]),
                        history=[str(i) for i in (example.get("history") or [])],
                        valid=bool(valid),
                    )
                )
                progress.update(1)
        return results
