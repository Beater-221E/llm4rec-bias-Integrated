# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
# (trainer/utils.py absolute_recall_mrr_ndcg_for_ks)

"""Ranking metrics used by the official LRU retriever / ranker."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def absolute_recall_mrr_ndcg_for_ks(
    scores: torch.Tensor,
    labels: torch.Tensor,
    ks: list[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    labels_oh = F.one_hot(labels, num_classes=scores.size(1))
    answer_count = labels_oh.sum(1)
    labels_float = labels_oh.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics[f"Recall@{k}"] = (
            (
                hits.sum(1)
                / torch.min(
                    torch.tensor([k], device=labels.device),
                    labels_oh.sum(1).float(),
                )
            )
            .mean()
            .cpu()
            .item()
        )
        metrics[f"MRR@{k}"] = (
            (hits / torch.arange(1, k + 1, device=labels.device).unsqueeze(0))
            .sum(1)
            .mean()
            .cpu()
            .item()
        )
        position = torch.arange(2, 2 + k, device=hits.device)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights).sum(1)
        idcg = torch.tensor(
            [weights[: min(int(n), k)].sum() for n in answer_count],
            device=dcg.device,
        )
        metrics[f"NDCG@{k}"] = (dcg / idcg).mean().cpu().item()
    return metrics
