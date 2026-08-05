"""Decoder 抽象 —— 把"LLM 输出"变成"ranked item list"。

这是三条路线唯一的公共接口。加新路线只需要实现一个 Decoder，
bias 评测、wandb 记录、stage 编排全部自动复用，不用碰其它任何地方。

    MiniOneRec : ConstrainedBeamDecoder  — SID token → 前缀树约束 beam → item
    Rec-R1     : BM25QueryDecoder        — query 文本 → BM25 检索      → item
    DPO4Rec    : KnowledgeRerankerDecoder— 推理文本 → adaptor+reranker → item
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from llm4rec.eval.bias import RankedResult


class Decoder(ABC):
    """把模型输出归一成 ranked item list。"""

    name: str = "decoder"

    @abstractmethod
    def decode_batch(
        self,
        model: Any,
        tokenizer: Any,
        examples: Sequence[dict[str, Any]],
        *,
        top_k: int,
    ) -> list[RankedResult]:
        """对一批样本产出 ranked list。

        ``examples`` 里每条至少要有 ``user_id`` / ``prompt`` / ``target_item`` /
        ``history``，具体字段由各路线的数据构建器保证。

        实现必须保证：即使模型输出非法（SID 解析失败、query 为空等），
        也要返回一条 ``valid=False`` 的 ``RankedResult``，而不是跳过 ——
        跳过会让 ``valid_rate`` 和分母都算错。
        """

    def close(self) -> None:
        """释放解码器自己持有的资源（检索索引、reranker 权重等）。"""
        return None
