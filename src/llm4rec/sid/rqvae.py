"""RQ-VAE：物品 embedding → 3 层残差量化码 → Semantic ID。

对齐官方 MiniOneRec（``rq/models/rqvae.py``）：冻结 text encoder 编码
title+description，再用 3 层 RQ-VAE 量化。

相对朴素 VQ 的几个必要改进（移植自 mor-reproduce，都是踩过坑的）：
  * 量化前先 L2 归一化 → PCA 降维 → 再 L2 归一化
  * 码本用大样本 KMeans warm-start，而不是拿一个 mini-batch 初始化
  * 周期性用高残差样本重置死码（否则码本利用率会塌到个位数）
  * 逐层码本利用率日志 + 碰撞率早停

★ 已知风险：mor-reproduce 实测在 Amazon23 Industrial 上 RQ-VAE 的碰撞率
  高于 residual-kmeans，所以它默认切到了 rqkmeans。这里按官方设定用 rqvae，
  但 build 时会强制检查碰撞率，超过 ``sid.max_collision_rate`` 直接失败。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.beta = beta
        self.latent_dim = latent_dim

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

    # ------------------------------------------------------------ 初始化
    @torch.no_grad()
    def warm_start(self, batch: torch.Tensor, seed: int = 42) -> None:
        """用一大批样本做逐层 KMeans 初始化。"""
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
                # 样本数不足以填满码本时，复制已有中心并加噪补齐
                idx = torch.randint(0, k, (self.codebook_size - k,), device=batch.device)
                centers = torch.cat([centers, centers[idx] + 0.01 * torch.randn_like(centers[idx])])
            self.codebooks[layer].copy_(centers)
            residual = residual - centers[torch.cdist(residual, centers).argmin(dim=-1)]

    # ------------------------------------------------------------ 量化
    def quantize(self, z: torch.Tensor):
        residual = z
        indices: list[torch.Tensor] = []
        quantized_sum = torch.zeros_like(z)
        vq_loss = z.new_zeros(())
        for layer in range(self.num_layers):
            cb = self.codebooks[layer]
            idx = torch.cdist(residual, cb).argmin(dim=-1)
            quantized = cb[idx]
            vq_loss = (
                vq_loss
                + F.mse_loss(residual.detach(), quantized)
                + self.beta * F.mse_loss(residual, quantized.detach())
            )
            # straight-through：前向用量化值，反向梯度直通到 residual
            quantized_sum = quantized_sum + residual + (quantized - residual).detach()
            residual = residual - quantized
            indices.append(idx)
        return quantized_sum, torch.stack(indices, dim=1), vq_loss

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        z_q, indices, vq_loss = self.quantize(z)
        recon = self.decoder(z_q)
        recon_loss = F.mse_loss(recon, x)
        return recon_loss + vq_loss, recon_loss.detach(), vq_loss.detach(), indices

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantize(self.encoder(x))[1]

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
        """把没被用到的码本条目替换成拟合最差（残差范数最大）的样本。"""
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


# ---------------------------------------------------------------- 预处理


def apply_pca(emb: np.ndarray, pca_dim: int, seed: int = 42) -> tuple[np.ndarray, dict]:
    """L2 归一化 → PCA → L2 归一化。"""
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


# ---------------------------------------------------------------- 训练


def train_rqvae(
    features: np.ndarray,
    cfg: dict[str, Any],
    *,
    levels: int,
    codebook_size: int,
    seed: int = 42,
    device: str = "cuda:0",
    log: Any = print,
) -> tuple[RQVAE, np.ndarray]:
    """训练 RQ-VAE，返回 ``(模型, 每个物品的码)``。"""
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
    ).to(dev)

    warm_n = min(int(cfg.get("warm_start_size") or 8192), x.shape[0])
    warm_idx = torch.randperm(x.shape[0])[:warm_n]
    model.warm_start(x[warm_idx].to(dev), seed=seed)
    log(f"[rqvae] KMeans warm-start 完成（{warm_n} 样本）")

    batch_size = int(cfg.get("batch_size") or 2048)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr") or 3e-4))

    epochs = int(cfg.get("epochs") or 4000)
    log_every = int(cfg.get("log_every") or 50)
    dead_every = int(cfg.get("dead_code_every") or 100)
    patience = int(cfg.get("early_collision_patience") or 40)

    best_collision = 1.0
    stale = 0
    all_x = x.to(dev)

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

        if epoch % log_every == 0 or epoch == epochs:
            codes = _encode_all(model, all_x)
            collision = collision_rate(codes)
            used = [int((u > 0).sum()) for u in usage]
            log(
                f"[rqvae] epoch {epoch:5d} loss={total / max(n_batch, 1):.5f} "
                f"码本利用={used}/{codebook_size} 碰撞率={collision:.4f}"
            )
            if collision < best_collision - 1e-6:
                best_collision, stale = collision, 0
            else:
                stale += 1
            if collision == 0.0:
                log(f"[rqvae] 碰撞率归零，第 {epoch} 轮提前停止")
                break
            if stale >= patience:
                log(f"[rqvae] 碰撞率 {patience} 次检查无改善，提前停止")
                break

    return model, _encode_all(model, all_x)


@torch.no_grad()
def _encode_all(model: RQVAE, x: torch.Tensor, chunk: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, x.shape[0], chunk):
        out.append(model.encode_indices(x[start : start + chunk]).cpu().numpy())
    return np.concatenate(out, axis=0)


def collision_rate(codes: np.ndarray) -> float:
    """有多少比例的物品和别人撞了码。"""
    n = codes.shape[0]
    if n == 0:
        return 0.0
    unique = len({tuple(int(c) for c in row) for row in codes})
    return float((n - unique) / n)


# ------------------------------------------------------- residual k-means 分支


def train_residual_kmeans(
    features: np.ndarray,
    *,
    levels: int,
    codebook_size: int,
    seed: int = 42,
    log: Any = print,
) -> np.ndarray:
    """逐层 KMeans 残差量化（无神经网络）。

    官方也提供这一支（RQ-Kmeans）。在 Amazon23 Industrial 上 mor-reproduce
    实测它的碰撞率明显低于 RQ-VAE，所以留作可切换的备选。
    """
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


def enforce_unique_last_code(codes: np.ndarray, codebook_size: int) -> np.ndarray:
    """保住前 n-1 层的语义前缀，只在同前缀桶内重排最后一位来消除碰撞。"""
    from collections import defaultdict

    out = codes.copy()
    last = codes.shape[1] - 1
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(codes):
        groups[tuple(int(c) for c in row[:last])].append(i)
    for prefix, members in groups.items():
        if len(members) <= 1:
            continue
        members.sort(key=lambda i: (int(codes[i, last]), i))
        if len(members) > codebook_size:
            # 极罕见：桶比码本还大，溢出到相邻的上一层码
            for j, i in enumerate(members):
                out[i, last] = j % codebook_size
                out[i, last - 1] = (prefix[last - 1] + j // codebook_size) % codebook_size
        else:
            for j, i in enumerate(members):
                out[i, last] = j
    return out


def break_collisions_extra_level(codes: np.ndarray) -> np.ndarray:
    """加一位碰撞消解码：同码的物品用 0,1,2,… 区分。"""
    from collections import defaultdict

    seen: dict[tuple, int] = defaultdict(int)
    extra = np.zeros((codes.shape[0], 1), dtype=np.int64)
    for i, row in enumerate(codes):
        key = tuple(int(c) for c in row)
        extra[i, 0] = seen[key]
        seen[key] += 1
    return np.concatenate([codes, extra], axis=1)
