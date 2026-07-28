"""Residual k-means semantic ID construction (MiniOneRec / TIGER-style)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


@torch.no_grad()
def embed_texts(
    texts: list[str],
    *,
    encoder: str = DEFAULT_ENCODER,
    device: str = "cuda",
    batch_size: int = 64,
) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(encoder)
    model = AutoModel.from_pretrained(encoder).to(device).eval()
    out: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        enc = tok(
            texts[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        emb = (h * mask).sum(1) / mask.sum(1)
        out.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
    return np.concatenate(out)


def kmeans(
    X: np.ndarray, K: int, iters: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    C = X[rng.choice(len(X), size=K, replace=False)].copy()
    assign = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        assign = d.argmin(1)
        for k in range(K):
            m = assign == k
            C[k] = X[m].mean(0) if m.any() else X[rng.integers(len(X))]
    return C, assign


def residual_quantize(
    X: np.ndarray, levels: int, K: int, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    res = X.copy()
    codes = np.zeros((len(X), levels), dtype=int)
    for level in range(levels):
        C, a = kmeans(res, K, iters=50, rng=rng)
        codes[:, level] = a
        res = res - C[a]
    return codes


def break_collisions(codes: np.ndarray) -> np.ndarray:
    """Append one disambiguation level so every ID is unique."""
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i, c in enumerate(map(tuple, codes)):
        groups[c].append(i)
    extra = np.zeros(len(codes), dtype=int)
    for members in groups.values():
        for j, i in enumerate(members):
            extra[i] = j
    return np.concatenate([codes, extra[:, None]], axis=1)
