"""DPO4Rec 的重排模型 —— 同时充当 reward model。

论文 §IV-B：LLM 生成的推理文本经 PLM 编码 + Knowledge Adaptor 降维后，
与 ID-based 表征**拼接**，一起喂给传统 reranker。reranker 输出列表的
NDCG 就是这份推理文本的分数。

论文对比了三个 backbone：
    DLCM    (Ai et al. SIGIR'18) —— GRU 编码候选列表的上下文
    PRM     (Pei et al. RecSys'19) —— Transformer 编码 + 个性化预训练向量
    SetRank (Pang et al. SIGIR'20) —— 自注意力，排列等变

我们默认 PRM（论文里 DPO4Rec 增益最稳的一个），另外两个可切换。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeAdaptor(nn.Module):
    """把 PLM 编出来的推理文本向量降到能和 ID embedding 拼接的维度。

    论文：encoder 冻结，只训这个 adaptor。
    """

    def __init__(self, in_dim: int, out_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _BaseReranker(nn.Module):
    """候选列表 → 每个候选一个分数。"""

    def __init__(
        self,
        n_items: int,
        *,
        hidden_dim: int = 128,
        knowledge_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, hidden_dim, padding_idx=n_items)
        self.knowledge_dim = knowledge_dim
        # 拼接 [ID 表征 ; 推理知识向量] → 投回 hidden_dim
        self.fuse = nn.Linear(hidden_dim + knowledge_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _embed(self, candidates: torch.Tensor, knowledge: torch.Tensor | None) -> torch.Tensor:
        """candidates [B, L] → [B, L, H]，可选地融合知识向量。"""
        x = self.item_emb(candidates)
        if knowledge is None:
            knowledge = x.new_zeros(x.shape[0], self.knowledge_dim)
        # 知识向量是 user 级的，广播到列表里每个候选上
        k = knowledge.unsqueeze(1).expand(-1, x.shape[1], -1)
        return self.dropout(F.relu(self.fuse(torch.cat([x, k], dim=-1))))

    def forward(
        self, candidates: torch.Tensor, knowledge: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise NotImplementedError


class PRM(_BaseReranker):
    """Transformer 编码器建模候选之间的相互影响（Pei et al. 2019）。"""

    def __init__(
        self,
        n_items: int,
        *,
        hidden_dim: int = 128,
        knowledge_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 64,
    ) -> None:
        super().__init__(
            n_items, hidden_dim=hidden_dim, knowledge_dim=knowledge_dim, dropout=dropout
        )
        # PRM 的关键：有位置编码（初始排序的位置本身是信号）
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self, candidates: torch.Tensor, knowledge: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self._embed(candidates, knowledge)
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.pos_emb(positions).unsqueeze(0)
        return self.score_head(self.encoder(x)).squeeze(-1)


class SetRank(_BaseReranker):
    """自注意力、排列等变 —— 没有位置编码（Pang et al. 2020）。"""

    def __init__(
        self,
        n_items: int,
        *,
        hidden_dim: int = 128,
        knowledge_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            n_items, hidden_dim=hidden_dim, knowledge_dim=knowledge_dim, dropout=dropout
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self, candidates: torch.Tensor, knowledge: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self._embed(candidates, knowledge)
        return self.score_head(self.encoder(x)).squeeze(-1)


class DLCM(_BaseReranker):
    """GRU 顺序编码候选列表（Ai et al. 2018）。"""

    def __init__(
        self,
        n_items: int,
        *,
        hidden_dim: int = 128,
        knowledge_dim: int = 64,
        dropout: float = 0.1,
        **_: Any,
    ) -> None:
        super().__init__(
            n_items, hidden_dim=hidden_dim, knowledge_dim=knowledge_dim, dropout=dropout
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.local_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self, candidates: torch.Tensor, knowledge: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self._embed(candidates, knowledge)
        seq, final = self.gru(x)
        # DLCM：每个候选和"整个列表的全局状态"交互后再打分
        global_state = final[-1].unsqueeze(1).expand(-1, seq.shape[1], -1)
        fused = torch.tanh(self.local_proj(torch.cat([seq, global_state], dim=-1)))
        return self.score_head(fused).squeeze(-1)


RERANKERS = {"prm": PRM, "setrank": SetRank, "dlcm": DLCM}


def build_reranker(kind: str, n_items: int, cfg: dict[str, Any]) -> _BaseReranker:
    key = str(kind).lower()
    if key not in RERANKERS:
        raise ValueError(f"未知 reranker '{kind}'，可用：{sorted(RERANKERS)}")
    return RERANKERS[key](
        n_items,
        hidden_dim=int(cfg.get("hidden_dim") or 128),
        knowledge_dim=int((cfg.get("knowledge_adaptor") or {}).get("output_dim") or 64),
        num_heads=int(cfg.get("num_heads") or 4),
        num_layers=int(cfg.get("num_layers") or 2),
        dropout=float(cfg.get("dropout") or 0.1),
    )


def listwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """ListNet 风格的交叉熵：目标物品应当得分最高。

    ``labels`` 是 one-hot（[B, L]，目标位置为 1）。
    """
    log_probs = F.log_softmax(scores, dim=-1)
    return -(labels * log_probs).sum(dim=-1).mean()
