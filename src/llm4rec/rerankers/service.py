"""Reranker 服务：训练 + 给推理文本打分（DPO4Rec 的 reward model）。

论文 Algorithm 1 里 ``Rec.evaluate(response_i)`` 就是这里的
``score_reasoning`` —— 把一份推理文本喂进 reranker，用重排后列表的 NDCG
当这份文本的分数。
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from llm4rec.core.exceptions import MissingArtifactError
from llm4rec.rerankers.models import KnowledgeAdaptor, build_reranker, listwise_loss


class ReasoningEncoder:
    """冻结的 PLM 文本编码器（论文里的 "text encoder"）。"""

    def __init__(self, model_name: str, *, max_length: int = 512, device: str = "cuda") -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.max_length = int(max_length)
        self.dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        hidden = self.model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


class RerankerService:
    """封装 reranker + adaptor + encoder，对外提供训练和打分。"""

    def __init__(
        self,
        cfg: dict[str, Any],
        item_ids: Sequence[str],
        *,
        device: str = "cuda",
    ) -> None:
        decoder_cfg = (cfg.get("decoder") or {})
        self.rr_cfg = decoder_cfg.get("reranker") or {}
        self.ka_cfg = decoder_cfg.get("knowledge_adaptor") or {}

        self.device = device if torch.cuda.is_available() else "cpu"
        self.item_ids = list(item_ids)
        self.index_of = {item: i for i, item in enumerate(self.item_ids)}
        self.pad_index = len(self.item_ids)
        self.candidate_size = int(self.rr_cfg.get("candidate_size") or 20)

        merged = dict(self.rr_cfg)
        merged["knowledge_adaptor"] = self.ka_cfg
        self.model = build_reranker(
            str(self.rr_cfg.get("kind") or "prm"), len(self.item_ids), merged
        ).to(self.device)

        self.encoder: ReasoningEncoder | None = None
        self.adaptor: KnowledgeAdaptor | None = None

    # -------------------------------------------------------------- 知识侧
    def ensure_encoder(self) -> None:
        """延迟加载 —— 训 reranker 的第一轮还没有推理文本，不必占显存。"""
        if self.encoder is not None:
            return
        name = str(self.ka_cfg.get("encoder") or "sentence-transformers/all-MiniLM-L6-v2")
        self.encoder = ReasoningEncoder(
            name,
            max_length=int(self.ka_cfg.get("max_length") or 512),
            device=self.device,
        )
        self.adaptor = KnowledgeAdaptor(
            self.encoder.dim, int(self.ka_cfg.get("output_dim") or 64)
        ).to(self.device)

    def _knowledge_vector(self, texts: Sequence[str] | None) -> torch.Tensor | None:
        if not texts:
            return None
        self.ensure_encoder()
        assert self.encoder is not None and self.adaptor is not None
        return self.adaptor(self.encoder.encode(texts))

    # ---------------------------------------------------------- 候选列表构建
    def build_candidates(
        self,
        example: dict[str, Any],
        popularity: dict[str, int],
        rng: random.Random,
    ) -> tuple[list[str], int]:
        """目标物品 + 按流行度采样的负例，打乱后返回 ``(候选列表, 目标位置)``。

        ★ 负例按流行度采样（不是均匀）：这是 re-ranking 的标准做法，也是
          bias 研究里必须固定的一个混杂因素 —— 候选集本身的流行度分布会
          直接影响 pop_lift 的读数。
        """
        target = str(example["target_item"])
        history = set(map(str, example.get("history") or []))
        pool = [i for i in self.item_ids if i != target and i not in history]
        weights = [float(popularity.get(i, 0)) + 1.0 for i in pool]

        n_neg = self.candidate_size - 1
        negatives: list[str] = []
        seen: set[str] = set()
        # 按权重放回抽样，去重后补齐
        for _ in range(n_neg * 4):
            if len(negatives) >= n_neg:
                break
            pick = rng.choices(pool, weights=weights, k=1)[0]
            if pick not in seen:
                seen.add(pick)
                negatives.append(pick)
        while len(negatives) < n_neg and pool:
            pick = rng.choice(pool)
            if pick not in seen:
                seen.add(pick)
                negatives.append(pick)

        candidates = [target, *negatives]
        rng.shuffle(candidates)
        return candidates, candidates.index(target)

    def _to_tensor(self, candidates: Sequence[str]) -> torch.Tensor:
        idx = [self.index_of.get(str(c), self.pad_index) for c in candidates]
        return torch.tensor([idx], dtype=torch.long, device=self.device)

    # ------------------------------------------------------------ 训练
    def train(
        self,
        examples: Sequence[dict[str, Any]],
        popularity: dict[str, int],
        *,
        logger: Any,
        seed: int = 42,
        reasoning_by_user: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """训 reranker。第一轮没有推理文本（纯 ID baseline），
        后续 DPO 迭代里可以把当前最好的推理文本喂进来一起训。"""
        epochs = int(self.rr_cfg.get("epochs") or 20)
        lr = float(self.rr_cfg.get("learning_rate") or 1e-3)
        batch_size = int(self.rr_cfg.get("batch_size") or 256)

        params: list[nn.Parameter] = list(self.model.parameters())
        if reasoning_by_user:
            self.ensure_encoder()
            assert self.adaptor is not None
            params += list(self.adaptor.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)

        rng = random.Random(seed)
        # 候选集固定住：跨 epoch 复用同一份，避免负例噪声淹没信号
        prepared = [
            (ex, *self.build_candidates(ex, popularity, rng)) for ex in examples
        ]

        self.model.train()
        for epoch in range(1, epochs + 1):
            rng.shuffle(prepared)
            total, n = 0.0, 0
            for start in range(0, len(prepared), batch_size):
                chunk = prepared[start : start + batch_size]
                cand = torch.tensor(
                    [[self.index_of.get(c, self.pad_index) for c in cands] for _, cands, _ in chunk],
                    dtype=torch.long,
                    device=self.device,
                )
                labels = torch.zeros(cand.shape, device=self.device)
                for row, (_, _, pos) in enumerate(chunk):
                    labels[row, pos] = 1.0

                knowledge = None
                if reasoning_by_user:
                    texts = [
                        reasoning_by_user.get(str(ex.get("user_id") or ""), "")
                        for ex, _, _ in chunk
                    ]
                    if any(texts):
                        knowledge = self._knowledge_vector(texts)

                scores = self.model(cand, knowledge)
                loss = listwise_loss(scores, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.item())
                n += 1

            if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
                logger.log_metrics(
                    {"reranker_loss": total / max(n, 1)},
                    stage="train_reranker",
                    step=epoch,
                    split="train",
                    wandb_prefix="train",
                )
                logger.info(f"[reranker] epoch {epoch}/{epochs} loss={total / max(n, 1):.4f}")

        return {"stage": "train_reranker", "epochs": epochs, "n_examples": len(examples)}

    # ------------------------------------------------------------ 打分
    @torch.no_grad()
    def score_reasoning(
        self, example: dict[str, Any], reasoning: str, *, top_k: int = 5
    ) -> float:
        """论文 Algorithm 1 的 ``Rec.evaluate(response)``：
        用这份推理文本重排候选列表，返回 NDCG@k。"""
        candidates = example.get("_candidates")
        target_pos = example.get("_target_pos")
        if candidates is None or target_pos is None:
            raise MissingArtifactError(
                "样本缺少 _candidates/_target_pos —— DPO 阶段前要先构建候选列表"
            )
        self.model.eval()
        knowledge = self._knowledge_vector([reasoning])
        scores = self.model(self._to_tensor(candidates), knowledge)[0]
        order = torch.argsort(scores, descending=True).tolist()
        rank = order.index(int(target_pos))
        return float(1.0 / math.log2(rank + 2)) if rank < top_k else 0.0

    @torch.no_grad()
    def rerank(
        self, example: dict[str, Any], reasoning: str | None, *, top_k: int
    ) -> list[str]:
        """返回重排后的 top-K item id（给 bias evaluator 用）。"""
        candidates = example["_candidates"]
        self.model.eval()
        knowledge = self._knowledge_vector([reasoning]) if reasoning else None
        scores = self.model(self._to_tensor(candidates), knowledge)[0]
        order = torch.argsort(scores, descending=True).tolist()
        return [str(candidates[i]) for i in order[:top_k]]

    # ------------------------------------------------------------ 存取
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "reranker": self.model.state_dict(),
            "item_ids": self.item_ids,
            "kind": str(self.rr_cfg.get("kind") or "prm"),
        }
        if self.adaptor is not None:
            payload["adaptor"] = self.adaptor.state_dict()
        torch.save(payload, path / "reranker.pt")

    def load(self, path: Path) -> None:
        blob = torch.load(Path(path) / "reranker.pt", map_location=self.device)
        self.model.load_state_dict(blob["reranker"])
        if "adaptor" in blob:
            self.ensure_encoder()
            assert self.adaptor is not None
            self.adaptor.load_state_dict(blob["adaptor"])
