"""RQ-VAE：物品 embedding → 3 层残差量化码 → Semantic ID（integrated / simplified）.

For MiniOneRec *reproduction* mode, use
``llm4rec.sid.minionerec_rqvae`` instead — that path ports the official
architecture (no PCA, layers=[2048..64], e_dim=32, codebooks=[256,256,256]).

This module keeps the generalized/simplified RQ-VAE used by ``mode: integrated``
experiments (optional PCA, compact MLP, dead-code reset, etc.).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm4rec.sid.base import collision_groups, collision_rate
from llm4rec.sid.collision import enforce_unique_last_code, resolve_collisions_sinkhorn

__all__ = [
    "RQVAE",
    "apply_pca",
    "train_rqvae",
    "train_residual_kmeans",
    "collision_rate",
    "collision_groups",
    "resolve_collisions_sinkhorn",
    "enforce_unique_last_code",
]


def sinkhorn(logits: torch.Tensor, epsilon: float, iters: int = 50) -> torch.Tensor:
    """Sinkhorn-Knopp soft assignment over the last dim (codebook).

    ``logits`` is ``[B, K]`` (typically ``-distance``). Same algorithm as
    MiniOneRec ``sinkhorn_algorithm``, but takes logits instead of distances.
    """
    q = torch.exp(logits / epsilon)
    b, k = q.shape
    q = q / q.sum().clamp(min=1e-12)
    for _ in range(iters):
        q = q / q.sum(dim=1, keepdim=True).clamp(min=1e-12)
        q = q / b
        q = q / q.sum(dim=0, keepdim=True).clamp(min=1e-12)
        q = q / k
    return q * b


class RQVAE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        *,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
        codebook_size: int = 256,
        beta: float = 0.25,
        kmeans_init: bool = True,
        kmeans_iters: int = 100,
        sk_iters: int = 50,
        sk_epsilons: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.beta = beta
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_iters = sk_iters
        self.sk_epsilons = list(
            sk_epsilons if sk_epsilons is not None else [0.0] * num_layers
        )

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )
        self.codebooks = nn.ParameterList(
            [
                nn.Parameter(torch.randn(codebook_size, latent_dim) * 0.01)
                for _ in range(num_layers)
            ]
        )

    @torch.no_grad()
    def warm_start(self, batch: torch.Tensor, seed: int = 42) -> None:
        from sklearn.cluster import KMeans

        residual = self.encoder(batch)
        n = residual.size(0)
        for layer in range(self.num_layers):
            k = min(self.codebook_size, n)
            km = KMeans(n_clusters=k, n_init=10, random_state=seed + layer)
            km.fit(residual.detach().cpu().numpy())
            centers = torch.tensor(
                km.cluster_centers_, device=batch.device, dtype=batch.dtype
            )
            if k < self.codebook_size:
                idx = torch.randint(0, k, (self.codebook_size - k,), device=batch.device)
                centers = torch.cat(
                    [centers, centers[idx] + 0.01 * torch.randn_like(centers[idx])]
                )
            self.codebooks[layer].copy_(centers)
            residual = residual - centers[torch.cdist(residual, centers).argmin(dim=-1)]

    def quantize(self, z: torch.Tensor, use_sk: bool = False):
        residual = z
        indices: list[torch.Tensor] = []
        quantized_sum = torch.zeros_like(z)
        vq_loss = z.new_zeros(())
        for layer in range(self.num_layers):
            cb = self.codebooks[layer]
            dist = torch.cdist(residual, cb)
            if use_sk and self.sk_epsilons[layer] > 0.0:
                idx = sinkhorn(-dist, self.sk_epsilons[layer], self.sk_iters).argmax(dim=-1)
            else:
                idx = dist.argmin(dim=-1)
            quantized = cb[idx]
            vq_loss = (
                vq_loss
                + F.mse_loss(residual.detach(), quantized)
                + self.beta * F.mse_loss(residual, quantized.detach())
            )
            quantized_sum = quantized_sum + residual + (quantized - residual).detach()
            residual = residual - quantized
            indices.append(idx)
        return quantized_sum, torch.stack(indices, dim=1), vq_loss

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        z_q, indices, vq_loss = self.quantize(z, use_sk=False)
        recon = self.decoder(z_q)
        recon_loss = F.mse_loss(recon, x)
        return recon_loss + vq_loss, recon_loss.detach(), vq_loss.detach(), indices

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor, use_sk: bool = False) -> torch.Tensor:
        return self.quantize(self.encoder(x), use_sk=use_sk)[1]

    @torch.no_grad()
    def layer_residuals(self, x: torch.Tensor) -> list[torch.Tensor]:
        residual = self.encoder(x)
        out = []
        for layer in range(self.num_layers):
            out.append(residual.clone())
            cb = self.codebooks[layer]
            residual = residual - cb[torch.cdist(residual, cb).argmin(dim=-1)]
        return out

    @torch.no_grad()
    def reset_dead_codes(
        self, x: torch.Tensor, usage: list[np.ndarray], min_count: int = 1
    ) -> list[int]:
        residuals = self.layer_residuals(x)
        resets = []
        for layer, (res, counts) in enumerate(zip(residuals, usage)):
            dead = np.where(counts < min_count)[0]
            if len(dead) == 0:
                resets.append(0)
                continue
            order = torch.argsort(res.norm(dim=-1), descending=True)
            pick = order[: max(len(dead), 1)]
            for i, code_id in enumerate(dead):
                src = res[pick[i % len(pick)]]
                self.codebooks[layer][int(code_id)].copy_(src + 0.01 * torch.randn_like(src))
            resets.append(len(dead))
        return resets


def apply_pca(emb: np.ndarray, pca_dim: int, seed: int = 42) -> tuple[np.ndarray, dict]:
    """L2 归一化 → PCA → L2 归一化（integrated mode only）。"""
    from sklearn.decomposition import PCA

    norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)
    normed = emb / norms
    dim = min(int(pca_dim), normed.shape[0], normed.shape[1])
    pca = PCA(n_components=dim, random_state=seed)
    z = pca.fit_transform(normed)
    z = z / np.linalg.norm(z, axis=1, keepdims=True).clip(min=1e-8)
    return z.astype(np.float32), {
        "pca_dim": dim,
        "original_dim": int(emb.shape[1]),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }


def train_rqvae(
    features: np.ndarray,
    cfg: dict[str, Any],
    *,
    levels: int,
    codebook_size: int,
    seed: int = 42,
    device: str = "cuda:0",
    out_dir: Any = None,
    log: Any = print,
) -> tuple[RQVAE, np.ndarray]:
    """Train simplified/integrated RQ-VAE."""
    from pathlib import Path

    from torch.utils.data import DataLoader, TensorDataset

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    x = torch.tensor(features, dtype=torch.float32)
    model = RQVAE(
        in_dim=x.shape[1],
        latent_dim=int(cfg.get("latent_dim") or 64),
        hidden_dim=int(cfg.get("hidden_dim") or 256),
        num_layers=levels,
        codebook_size=codebook_size,
        beta=float(cfg.get("beta") or 0.25),
        kmeans_init=bool(cfg.get("kmeans_init", True)),
        kmeans_iters=int(cfg.get("kmeans_iters") or 100),
        sk_iters=int(cfg.get("sk_iters") or 50),
        sk_epsilons=list(cfg.get("sk_epsilons") or [0.0] * levels),
    ).to(dev)

    warm_n = min(int(cfg.get("warm_start_size") or 8192), x.shape[0])
    warm_idx = torch.randperm(x.shape[0])[:warm_n]
    model.warm_start(x[warm_idx].to(dev), seed=seed)
    log(f"[rqvae] KMeans warm-start 完成（{warm_n} 样本）")

    batch_size = int(cfg.get("batch_size") or 2048)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr") or 3e-4))

    epochs = int(cfg.get("epochs") or 2000)
    log_every = int(cfg.get("log_every") or 50)
    dead_every = int(cfg.get("dead_code_every") or 100)
    patience = int(cfg.get("early_collision_patience") or 40)

    out_dir = Path(out_dir) if out_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_collision = 1.0
    stale = 0
    all_x = x.to(dev)

    def _ckpt(path: Path, epoch_no: int, metric: float) -> None:
        torch.save(
            {
                "epoch": epoch_no,
                "metric": metric,
                "state_dict": model.state_dict(),
                "config": {
                    "in_dim": x.shape[1],
                    "latent_dim": model.latent_dim,
                    "hidden_dim": model.hidden_dim,
                    "num_layers": model.num_layers,
                    "codebook_size": model.codebook_size,
                    "beta": model.beta,
                    "implementation": "integrated_simplified",
                },
            },
            path,
        )

    for epoch in range(1, epochs + 1):
        model.train()
        total, n_batch = 0.0, 0
        usage = [np.zeros(codebook_size, dtype=np.int64) for _ in range(levels)]
        for (batch,) in loader:
            batch = batch.to(dev)
            loss, recon, vq, indices = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            n_batch += 1
            idx_np = indices.detach().cpu().numpy()
            for layer in range(levels):
                np.add.at(usage[layer], idx_np[:, layer], 1)

        if epoch % dead_every == 0:
            sample = all_x[torch.randperm(all_x.shape[0])[: min(4096, all_x.shape[0])]]
            resets = model.reset_dead_codes(sample, usage)
            if sum(resets) and epoch % log_every == 0:
                log(f"[rqvae] epoch {epoch} 重置死码 {resets}")

        epoch_loss = total / max(n_batch, 1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            if out_dir is not None:
                _ckpt(out_dir / "best_loss_model.pth", epoch, best_loss)

        if epoch % log_every == 0 or epoch == epochs:
            codes = _encode_all(model, all_x)
            collision = collision_rate(codes)
            used = [int((u > 0).sum()) for u in usage]
            log(
                f"[rqvae] epoch {epoch:5d} loss={epoch_loss:.5f} "
                f"码本利用={used}/{codebook_size} 碰撞率={collision:.4f}"
            )
            if collision < best_collision - 1e-6:
                best_collision, stale = collision, 0
                if out_dir is not None:
                    _ckpt(out_dir / "best_collision_model.pth", epoch, best_collision)
            else:
                stale += 1
            if collision == 0.0:
                log(f"[rqvae] 碰撞率归零，第 {epoch} 轮提前停止")
                break
            if stale >= patience:
                log(f"[rqvae] 碰撞率 {patience} 次检查无改善，提前停止")
                break

    if out_dir is not None and (out_dir / "best_collision_model.pth").exists():
        ckpt = torch.load(
            out_dir / "best_collision_model.pth", map_location=dev, weights_only=False
        )
        model.load_state_dict(ckpt["state_dict"])
        log(
            f"[rqvae] 已加载 best_collision_model"
            f"（epoch={ckpt['epoch']} 碰撞率={ckpt['metric']:.4f}）"
        )

    return model, _encode_all(model, all_x)


@torch.no_grad()
def _encode_all(model: RQVAE, x: torch.Tensor, chunk: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, x.shape[0], chunk):
        out.append(model.encode_indices(x[start : start + chunk]).cpu().numpy())
    return np.concatenate(out, axis=0)


def train_residual_kmeans(
    features: np.ndarray,
    *,
    levels: int,
    codebook_size: int,
    seed: int = 42,
    log: Any = print,
) -> np.ndarray:
    from sklearn.cluster import KMeans

    residual = features.astype(np.float32).copy()
    codes = np.zeros((features.shape[0], levels), dtype=np.int64)
    for layer in range(levels):
        k = min(codebook_size, residual.shape[0])
        km = KMeans(n_clusters=k, n_init=10, random_state=seed + layer)
        assign = km.fit_predict(residual)
        codes[:, layer] = assign
        residual = residual - km.cluster_centers_[assign]
        log(f"[rqkmeans] 第 {layer} 层完成，使用了 {len(set(assign))}/{codebook_size} 个码")
    return codes
