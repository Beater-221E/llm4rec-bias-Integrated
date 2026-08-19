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
from llm4rec.tracking.progress import overwrite_progress


def _ndcg_at_k(order: Sequence[int], relevant: set[int], k: int) -> float:
    """Binary NDCG@k; one relevant item reduces to 1/log2(rank+2) if in top-k."""
    if k <= 0 or not relevant:
        return 0.0
    dcg = 0.0
    for rank, idx in enumerate(order[:k]):
        if int(idx) in relevant:
            dcg += 1.0 / math.log2(rank + 2)
    n_rel = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return float(dcg / idcg) if idcg else 0.0


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
        # KAR ``rerank_list_len=10`` / ``rerank_item_from_hist=4``
        # (DPO4Rec §V-A-4 follows KAR settings for the reward-model recommender).
        self.candidate_size = int(self.rr_cfg.get("candidate_size") or 10)
        self.n_positives = int(self.rr_cfg.get("n_positives") or 4)

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
    def _positives(self, example: dict[str, Any]) -> list[str]:
        """KAR: up to ``n_positives`` subsequent interactions are relevant.

        ``target_item`` is always kept first so bias eval still has a primary item.
        """
        ordered: list[str] = []
        seen: set[str] = set()
        for raw in (example.get("positive_items") or [example["target_item"]]):
            item = str(raw)
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
            if len(ordered) >= self.n_positives:
                break
        target = str(example["target_item"])
        if target not in seen:
            ordered = [target, *ordered][: self.n_positives]
        elif ordered[0] != target:
            ordered = [target, *[item for item in ordered if item != target]]
        return ordered

    def build_candidates(
        self,
        example: dict[str, Any],
        popularity: dict[str, int],
        rng: random.Random,
    ) -> tuple[list[str], int, list[int]]:
        """KAR/DPO4Rec candidate list: positives + uniform unobserved negatives.

        Paper §V-A-3: rerank top-K drawn from items the user has not interacted
        with. KAR ``generate_rerank_data`` uses ``random.sample`` (uniform) to
        fill ``rerank_list_len`` after taking the next ``rerank_item_from_hist``
        interactions as positives. ``popularity`` is unused; kept for call-site
        compatibility.
        """
        del popularity
        positives = self._positives(example)
        history = {str(item) for item in (example.get("history") or [])}
        exclude = history | set(positives)
        n_neg = max(0, self.candidate_size - len(positives))

        negatives: list[str] = []
        seen = set(exclude)
        n_items = len(self.item_ids)
        max_tries = max(n_neg * 16, 64)
        for _ in range(max_tries):
            if len(negatives) >= n_neg or n_items == 0:
                break
            pick = self.item_ids[rng.randrange(n_items)]
            if pick in seen:
                continue
            seen.add(pick)
            negatives.append(pick)
        if len(negatives) < n_neg:
            for item in self.item_ids:
                if len(negatives) >= n_neg:
                    break
                if item not in seen:
                    seen.add(item)
                    negatives.append(item)

        candidates = [*positives, *negatives]
        rng.shuffle(candidates)
        pos_indices = [candidates.index(item) for item in positives]
        return candidates, candidates.index(str(example["target_item"])), pos_indices

    def assign_candidates(
        self,
        examples: Sequence[dict[str, Any]],
        popularity: dict[str, int],
        rng: random.Random,
        *,
        desc: str = "reranker/candidates",
        logger: Any = None,
    ) -> None:
        """In-place attach ``_candidates`` / ``_target_pos`` / ``_pos_indices``.

        Already-filled examples are skipped so ``train()`` can reuse the lists
        built in ``build_context`` instead of sampling a second time.
        """
        if not examples:
            return
        total = len(examples)
        _ = logger
        with overwrite_progress(total, desc, log_interval_s=5.0) as bar:
            for example in examples:
                missing = (
                    example.get("_candidates") is None
                    or example.get("_target_pos") is None
                    or example.get("_pos_indices") is None
                )
                if missing:
                    candidates, pos, pos_indices = self.build_candidates(
                        example, popularity, rng
                    )
                    example["_candidates"] = candidates
                    example["_target_pos"] = pos
                    example["_pos_indices"] = pos_indices
                bar.update(1)

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
        self.ensure_device()
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
        # 候选集固定住：优先复用 build_context 已经采好的列表，避免再扫一遍全库
        self.assign_candidates(
            examples, popularity, rng, desc="reranker/candidates", logger=logger
        )
        prepared = [
            (ex, ex["_candidates"], int(ex["_target_pos"])) for ex in examples
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
                for row, (ex, _, _) in enumerate(chunk):
                    for pos in ex.get("_pos_indices") or [ex["_target_pos"]]:
                        labels[row, int(pos)] = 1.0
                labels = labels / labels.sum(dim=-1, keepdim=True).clamp(min=1e-6)

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
        用这份推理文本重排候选列表，返回 NDCG@k（§V-A-3）。"""
        candidates = example.get("_candidates")
        target_pos = example.get("_target_pos")
        if candidates is None or target_pos is None:
            raise MissingArtifactError(
                "样本缺少 _candidates/_target_pos —— DPO 阶段前要先构建候选列表"
            )
        relevant = {
            int(p) for p in (example.get("_pos_indices") or [target_pos])
        }
        self.ensure_device()
        self.model.eval()
        knowledge = self._knowledge_vector([reasoning])
        scores = self.model(self._to_tensor(candidates), knowledge)[0]
        order = torch.argsort(scores, descending=True).tolist()
        return _ndcg_at_k(order, relevant, top_k)

    @torch.no_grad()
    def rerank(
        self, example: dict[str, Any], reasoning: str | None, *, top_k: int
    ) -> list[str]:
        """返回重排后的 top-K item id（给 bias evaluator 用）。"""
        candidates = example["_candidates"]
        self.ensure_device()
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

    def release_cuda(self) -> None:
        """Park the reranker on CPU so SFT can use the full GPU."""
        self.model.to("cpu")
        if self.encoder is not None:
            self.encoder.model.to("cpu")
            self.encoder.device = "cpu"
        if self.adaptor is not None:
            self.adaptor.to("cpu")
        self.device = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure_device(self, device: str | None = None) -> None:
        target = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available() and str(target).startswith("cuda"):
            target = "cpu"
        self.device = target
        self.model.to(self.device)
        if self.encoder is not None:
            self.encoder.model.to(self.device)
            self.encoder.device = self.device
        if self.adaptor is not None:
            self.adaptor.to(self.device)
