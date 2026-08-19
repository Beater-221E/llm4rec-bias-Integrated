"""DPO4Rec 路线的 Decoder：推理文本 → adaptor + reranker → ranked item list。

★ 和另外两条路线的关键差别：DPO4Rec 是 **re-ranking**，LLM 不直接产出物品。
  所以它的 ranked list 是从一个**候选集**里排出来的，不是全库。

  这对 bias 指标的解读有直接影响：
    * ``coverage`` / ``exposure_gini`` 受候选集构造方式支配，
      跨路线比较时要记住 DPO4Rec 的分母是候选集不是全库；
    * ``pop_lift`` / ``tier_hr`` 仍然可比 —— 它们只看被排上来的物品是什么。
  候选集对齐 KAR / DPO4Rec §V-A：后续交互作正例，未交互物品均匀采负例，
  固定 seed，保证跨 step / 跨 run 可比。
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from llm4rec.decoders.base import Decoder
from llm4rec.eval.bias import RankedResult
from llm4rec.tracking.progress import overwrite_progress


class KnowledgeRerankerDecoder(Decoder):
    """LLM 生成一份推理文本 → 喂 reranker → 重排候选列表。"""

    name = "knowledge_reranker"

    def __init__(
        self,
        reranker_service: Any,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.service = reranker_service
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)

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
        if hasattr(self.service, "ensure_device"):
            self.service.ensure_device()
        device = next(model.parameters()).device
        pad = tokenizer.pad_token_id or tokenizer.eos_token_id

        results: list[RankedResult] = []
        n_total = len(examples)
        with overwrite_progress(
            n_total,
            "eval",
            global_total=progress_total,
            progress_dir=progress_dir,
            name=progress_name or "eval/knowledge_reranker",
        ) as progress:
            for example in examples:
                encoded = tokenizer.apply_chat_template(
                    example["prompt"], add_generation_prompt=True, return_tensors="pt"
                )
                ids = (
                    encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
                ).to(device)
                prompt_len = ids.shape[1]

                # 评测时用贪心，保证同一个 checkpoint 的读数可复现
                output = model.generate(
                    ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                    pad_token_id=pad,
                )
                reasoning = tokenizer.decode(
                    output[0, prompt_len:], skip_special_tokens=True
                ).strip()

                ranked = self.service.rerank(example, reasoning or None, top_k=top_k)
                results.append(
                    RankedResult(
                        user_id=str(example.get("user_id") or ""),
                        ranked_items=[str(i) for i in ranked],
                        target_item=str(example["target_item"]),
                        history=[str(i) for i in (example.get("history") or [])],
                        valid=bool(reasoning),
                    )
                )
                progress.update(1)
        return results
