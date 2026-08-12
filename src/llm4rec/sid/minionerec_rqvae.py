"""Official MiniOneRec RQ-VAE — ported from AkaliKong/MiniOneRec ``rq/models``.

Faithful reproduction of:
  * MLP encoder/decoder layers ``[2048, 1024, 512, 256, 128, 64]`` → ``e_dim=32``
  * 3-level residual VQ with codebook sizes ``[256, 256, 256]``
  * commitment / codebook loss with ``beta=0.25``
  * k-means codebook initialization
  * Sinkhorn-constrained assignment for collision resolution (last level)

No PCA is applied. Embeddings are consumed at their native dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------- layers


def activation_layer(activation_name: str = "relu") -> nn.Module | None:
    name = (activation_name or "relu").lower()
    if name in {"none", ""}:
        return None
    mapping = {
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
    }
    if name not in mapping:
        raise ValueError(f"unsupported activation: {activation_name}")
    return mapping[name]()


class MLPLayers(nn.Module):
    """Port of MiniOneRec ``rq/models/layers.MLPLayers``."""

    def __init__(
        self,
        layers: list[int],
        dropout: float = 0.0,
        activation: str = "relu",
        bn: bool = False,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        for idx, (inp, out) in enumerate(zip(layers[:-1], layers[1:])):
            modules.append(nn.Dropout(p=dropout))
            modules.append(nn.Linear(inp, out))
            if bn and idx != (len(layers) - 2):
                modules.append(nn.BatchNorm1d(num_features=out))
            act = activation_layer(activation)
            if act is not None and idx != (len(layers) - 2):
                modules.append(act)
        self.mlp_layers = nn.Sequential(*modules)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_layers(x)


def kmeans(samples: torch.Tensor, num_clusters: int, num_iters: int = 10) -> torch.Tensor:
    from sklearn.cluster import KMeans

    x = samples.detach().cpu().numpy()
    k = min(int(num_clusters), max(1, x.shape[0]))
    cluster = KMeans(n_clusters=k, max_iter=num_iters, n_init=10).fit(x)
    centers = torch.from_numpy(cluster.cluster_centers_).to(
        device=samples.device, dtype=samples.dtype
    )
    if k < num_clusters:
        # Pad by repeating centers with small noise (tiny batches / smoke tests)
        reps = []
        need = num_clusters - k
        idx = torch.randint(0, k, (need,), device=centers.device)
        reps = centers[idx] + 0.01 * torch.randn(need, centers.shape[1], device=centers.device, dtype=centers.dtype)
        centers = torch.cat([centers, reps], dim=0)
    return centers



@torch.no_grad()
def sinkhorn_algorithm(
    distances: torch.Tensor, epsilon: float, sinkhorn_iterations: int
) -> torch.Tensor:
    """Port of MiniOneRec ``rq/models/layers.sinkhorn_algorithm``."""
    q = torch.exp(-distances / epsilon)
    b = q.shape[0]
    k = q.shape[1]
    q = q / q.sum()
    for _ in range(sinkhorn_iterations):
        q = q / torch.sum(q, dim=1, keepdim=True)
        q = q / b
        q = q / torch.sum(q, dim=0, keepdim=True)
        q = q / k
    return q * b


# --------------------------------------------------------------------------- VQ


class VectorQuantizer(nn.Module):
    """Port of MiniOneRec ``rq/models/vq.VectorQuantizer``."""

    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float = 0.25,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        sk_epsilon: float = 0.003,
        sk_iters: int = 100,
    ) -> None:
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self) -> torch.Tensor:
        return self.embedding.weight

    def init_emb(self, data: torch.Tensor) -> None:
        centers = kmeans(data, self.n_e, self.kmeans_iters)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances: torch.Tensor) -> torch.Tensor:
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        return (distances - middle) / amplitude

    def forward(self, x: torch.Tensor, use_sk: bool = True):
        latent = x.view(-1, self.e_dim)
        if not self.initted and self.training:
            self.init_emb(latent)

        d = (
            torch.sum(latent**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(latent, self.embedding.weight.t())
        )
        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d_c = self.center_distance_for_constraint(d).double()
            q = sinkhorn_algorithm(d_c, self.sk_epsilon, self.sk_iters)
            if torch.isnan(q).any() or torch.isinf(q).any():
                indices = torch.argmin(d, dim=-1)
            else:
                indices = torch.argmax(q, dim=-1)

        x_q = self.embedding(indices).view(x.shape)
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss
        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices


class ResidualVectorQuantizer(nn.Module):
    """Port of MiniOneRec ``rq/models/rq.ResidualVectorQuantizer``."""

    def __init__(
        self,
        n_e_list: list[int],
        e_dim: int,
        sk_epsilons: list[float],
        beta: float = 0.25,
        kmeans_init: bool = False,
        kmeans_iters: int = 100,
        sk_iters: int = 100,
    ) -> None:
        super().__init__()
        self.n_e_list = list(n_e_list)
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.vq_layers = nn.ModuleList(
            [
                VectorQuantizer(
                    n_e,
                    e_dim,
                    beta=beta,
                    kmeans_init=kmeans_init,
                    kmeans_iters=kmeans_iters,
                    sk_epsilon=sk_epsilon,
                    sk_iters=sk_iters,
                )
                for n_e, sk_epsilon in zip(n_e_list, sk_epsilons)
            ]
        )

    def get_codebook(self) -> torch.Tensor:
        return torch.stack([q.get_codebook() for q in self.vq_layers])

    def forward(self, x: torch.Tensor, use_sk: bool = True):
        all_losses = []
        all_indices = []
        x_q = torch.zeros_like(x)
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res
            all_losses.append(loss)
            all_indices.append(indices)
        mean_losses = torch.stack(all_losses).mean()
        all_indices_t = torch.stack(all_indices, dim=-1)
        return x_q, mean_losses, all_indices_t


# --------------------------------------------------------------------------- RQ-VAE


class MiniOneRecRQVAE(nn.Module):
    """Port of MiniOneRec ``rq/models/rqvae.RQVAE``."""

    def __init__(
        self,
        in_dim: int,
        num_emb_list: list[int] | None = None,
        e_dim: int = 32,
        layers: list[int] | None = None,
        dropout_prob: float = 0.0,
        bn: bool = False,
        loss_type: str = "mse",
        quant_loss_weight: float = 1.0,
        beta: float = 0.25,
        kmeans_init: bool = True,
        kmeans_iters: int = 100,
        sk_epsilons: list[float] | None = None,
        sk_iters: int = 50,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_emb_list = list(num_emb_list or [256, 256, 256])
        self.e_dim = e_dim
        self.layers = list(layers or [2048, 1024, 512, 256, 128, 64])
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = list(sk_epsilons if sk_epsilons is not None else [0.0, 0.0, 0.0])
        self.sk_iters = sk_iters

        encode_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(encode_dims, dropout=self.dropout_prob, bn=self.bn)
        self.rq = ResidualVectorQuantizer(
            self.num_emb_list,
            self.e_dim,
            sk_epsilons=self.sk_epsilons,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_iters=self.sk_iters,
        )
        self.decoder = MLPLayers(encode_dims[::-1], dropout=self.dropout_prob, bn=self.bn)

    @property
    def num_layers(self) -> int:
        return len(self.num_emb_list)

    def forward(self, x: torch.Tensor, use_sk: bool = True):
        z = self.encoder(x)
        x_q, rq_loss, indices = self.rq(z, use_sk=use_sk)
        out = self.decoder(x_q)
        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs: torch.Tensor, use_sk: bool = False) -> torch.Tensor:
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices

    def compute_loss(self, out: torch.Tensor, quant_loss: torch.Tensor, xs: torch.Tensor):
        if self.loss_type == "mse":
            loss_recon = F.mse_loss(out, xs, reduction="mean")
        elif self.loss_type == "l1":
            loss_recon = F.l1_loss(out, xs, reduction="mean")
        else:
            raise ValueError(f"incompatible loss type: {self.loss_type}")
        loss_total = loss_recon + self.quant_loss_weight * quant_loss
        return loss_total, loss_recon


# --------------------------------------------------------------------------- defaults / config


MINIONEREC_RQVAE_DEFAULTS: dict[str, Any] = {
    "layers": [2048, 1024, 512, 256, 128, 64],
    "e_dim": 32,
    "num_emb_list": [256, 256, 256],
    "lr": 1e-3,
    "epochs": 10000,
    "batch_size": 20480,
    "beta": 0.25,
    "quant_loss_weight": 1.0,
    "loss_type": "mse",
    "dropout_prob": 0.0,
    "bn": False,
    "kmeans_init": True,
    "kmeans_iters": 100,
    "sk_epsilons": [0.0, 0.0, 0.0],
    "sk_iters": 50,
    "sk_epsilon_last": 0.003,
    "collision_retry_iters": 20,
    "eval_step": 50,
    "warmup_epochs": 50,
    "learner": "AdamW",
    "lr_scheduler_type": "constant",
    "weight_decay": 0.0,
    "pca_dim": None,  # official: no PCA
}


@dataclass
class MiniOneRecRQVAEConfig:
    layers: list[int] = field(default_factory=lambda: list(MINIONEREC_RQVAE_DEFAULTS["layers"]))
    e_dim: int = 32
    num_emb_list: list[int] = field(
        default_factory=lambda: list(MINIONEREC_RQVAE_DEFAULTS["num_emb_list"])
    )
    lr: float = 1e-3
    epochs: int = 10000
    batch_size: int = 20480
    beta: float = 0.25
    quant_loss_weight: float = 1.0
    loss_type: str = "mse"
    dropout_prob: float = 0.0
    bn: bool = False
    kmeans_init: bool = True
    kmeans_iters: int = 100
    sk_epsilons: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    sk_iters: int = 50
    sk_epsilon_last: float = 0.003
    collision_retry_iters: int = 20
    eval_step: int = 50
    warmup_epochs: int = 50
    learner: str = "AdamW"
    lr_scheduler_type: str = "constant"
    weight_decay: float = 0.0

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> MiniOneRecRQVAEConfig:
        raw = dict(MINIONEREC_RQVAE_DEFAULTS)
        if cfg:
            raw.update({k: v for k, v in cfg.items() if v is not None})
        # Accept integrated-style aliases
        if "latent_dim" in (cfg or {}) and "e_dim" not in (cfg or {}):
            raw["e_dim"] = int(cfg["latent_dim"])  # type: ignore[index]
        if "codebook_size" in (cfg or {}) and "num_emb_list" not in (cfg or {}):
            k = int(cfg["codebook_size"])  # type: ignore[index]
            levels = int((cfg or {}).get("levels") or len(raw["num_emb_list"]))
            raw["num_emb_list"] = [k] * levels
        return cls(
            layers=list(raw["layers"]),
            e_dim=int(raw["e_dim"]),
            num_emb_list=list(raw["num_emb_list"]),
            lr=float(raw["lr"]),
            epochs=int(raw["epochs"]),
            batch_size=int(raw["batch_size"]),
            beta=float(raw["beta"]),
            quant_loss_weight=float(raw["quant_loss_weight"]),
            loss_type=str(raw["loss_type"]),
            dropout_prob=float(raw["dropout_prob"]),
            bn=bool(raw["bn"]),
            kmeans_init=bool(raw["kmeans_init"]),
            kmeans_iters=int(raw["kmeans_iters"]),
            sk_epsilons=list(raw["sk_epsilons"]),
            sk_iters=int(raw["sk_iters"]),
            sk_epsilon_last=float(raw["sk_epsilon_last"]),
            collision_retry_iters=int(raw["collision_retry_iters"]),
            eval_step=int(raw["eval_step"]),
            warmup_epochs=int(raw["warmup_epochs"]),
            learner=str(raw["learner"]),
            lr_scheduler_type=str(raw["lr_scheduler_type"]),
            weight_decay=float(raw["weight_decay"]),
        )


def assert_minionerec_architecture(cfg: MiniOneRecRQVAEConfig) -> None:
    """Hard checks used by reproduction-mode preflight / tests."""
    if cfg.layers != [2048, 1024, 512, 256, 128, 64]:
        raise ValueError(f"official layers mismatch: {cfg.layers}")
    if cfg.e_dim != 32:
        raise ValueError(f"official e_dim mismatch: {cfg.e_dim}")
    if cfg.num_emb_list != [256, 256, 256]:
        raise ValueError(f"official codebook mismatch: {cfg.num_emb_list}")


# --------------------------------------------------------------------------- train + encode


def _build_optimizer(model: nn.Module, cfg: MiniOneRecRQVAEConfig):
    params = model.parameters()
    name = cfg.learner.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def _collision_rate_from_indices(indices: np.ndarray) -> float:
    n = indices.shape[0]
    if n == 0:
        return 0.0
    unique = len({tuple(int(c) for c in row) for row in indices})
    return float((n - unique) / n)


@torch.no_grad()
def encode_all_minionerec(
    model: MiniOneRecRQVAE, x: torch.Tensor, *, use_sk: bool = False, chunk: int = 8192
) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, x.shape[0], chunk):
        idx = model.get_indices(x[start : start + chunk], use_sk=use_sk)
        out.append(idx.view(-1, idx.shape[-1]).cpu().numpy())
    return np.concatenate(out, axis=0)


def train_minionerec_rqvae(
    features: np.ndarray,
    cfg: dict[str, Any] | MiniOneRecRQVAEConfig | None = None,
    *,
    seed: int = 2024,
    device: str = "cuda:0",
    out_dir: Path | str | None = None,
    log: Any = print,
) -> tuple[MiniOneRecRQVAE, np.ndarray]:
    """Train official-style RQ-VAE. Returns ``(model, codes)`` using best_collision ckpt."""
    ocfg = cfg if isinstance(cfg, MiniOneRecRQVAEConfig) else MiniOneRecRQVAEConfig.from_dict(cfg)
    raw_cfg = cfg if isinstance(cfg, dict) else {}
    if isinstance(raw_cfg, dict) and raw_cfg.get("pca_dim") not in (None, 0, False):
        raise ValueError(
            "official MiniOneRec RQ-VAE must not use PCA "
            f"(got pca_dim={raw_cfg.get('pca_dim')})"
        )

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    x = torch.tensor(features, dtype=torch.float32)
    model = MiniOneRecRQVAE(
        in_dim=x.shape[1],
        num_emb_list=ocfg.num_emb_list,
        e_dim=ocfg.e_dim,
        layers=ocfg.layers,
        dropout_prob=ocfg.dropout_prob,
        bn=ocfg.bn,
        loss_type=ocfg.loss_type,
        quant_loss_weight=ocfg.quant_loss_weight,
        beta=ocfg.beta,
        kmeans_init=ocfg.kmeans_init,
        kmeans_iters=ocfg.kmeans_iters,
        sk_epsilons=ocfg.sk_epsilons,
        sk_iters=ocfg.sk_iters,
    ).to(dev)

    batch_size = min(int(ocfg.batch_size), max(1, x.shape[0]))
    loader = DataLoader(
        TensorDataset(x), batch_size=batch_size, shuffle=True, drop_last=False
    )
    optimizer = _build_optimizer(model, ocfg)

    # Official uses transformers schedulers; keep a simple constant+warmup fallback
    steps_per_epoch = max(1, len(loader))
    warmup_steps = ocfg.warmup_epochs * steps_per_epoch
    try:
        from transformers import get_constant_schedule_with_warmup

        scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)
    except Exception:  # noqa: BLE001
        scheduler = None

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_collision = float("inf")
    all_x = x.to(dev)
    log(
        f"[official-rqvae] in_dim={x.shape[1]} layers={ocfg.layers} e_dim={ocfg.e_dim} "
        f"codebooks={ocfg.num_emb_list} lr={ocfg.lr} epochs={ocfg.epochs} "
        f"batch={batch_size} (no PCA)"
    )

    def _save(name: str, epoch: int, metric: float) -> None:
        if out_path is None:
            return
        torch.save(
            {
                "epoch": epoch,
                "metric": metric,
                "state_dict": model.state_dict(),
                "config": {
                    "in_dim": x.shape[1],
                    "num_emb_list": ocfg.num_emb_list,
                    "e_dim": ocfg.e_dim,
                    "layers": ocfg.layers,
                    "dropout_prob": ocfg.dropout_prob,
                    "bn": ocfg.bn,
                    "loss_type": ocfg.loss_type,
                    "quant_loss_weight": ocfg.quant_loss_weight,
                    "beta": ocfg.beta,
                    "kmeans_init": ocfg.kmeans_init,
                    "kmeans_iters": ocfg.kmeans_iters,
                    "sk_epsilons": ocfg.sk_epsilons,
                    "sk_iters": ocfg.sk_iters,
                    "implementation": "minionerec_reference",
                },
            },
            out_path / name,
        )

    for epoch in range(ocfg.epochs):
        model.train()
        total_loss = 0.0
        n_batch = 0
        for (batch,) in loader:
            batch = batch.to(dev)
            optimizer.zero_grad()
            out, rq_loss, _ = model(batch, use_sk=False)
            loss, _recon = model.compute_loss(out, rq_loss, xs=batch)
            if torch.isnan(loss):
                raise ValueError("official RQ-VAE training loss is nan")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += float(loss.item())
            n_batch += 1

        epoch_loss = total_loss / max(n_batch, 1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            _save("best_loss_model.pth", epoch, best_loss)

        if (epoch + 1) % max(1, ocfg.eval_step) == 0 or epoch + 1 == ocfg.epochs:
            codes = encode_all_minionerec(model, all_x, use_sk=False)
            collision = _collision_rate_from_indices(codes)
            log(
                f"[official-rqvae] epoch {epoch + 1}/{ocfg.epochs} "
                f"loss={epoch_loss:.5f} collision={collision:.4f}"
            )
            if collision < best_collision:
                best_collision = collision
                _save("best_collision_model.pth", epoch, best_collision)
            if collision == 0.0:
                log(f"[official-rqvae] collision reached 0 at epoch {epoch + 1}")
                break

        # Soft progress cap for smoke / tiny datasets: avoid 10k epochs on 8 items
        if x.shape[0] < 64 and epoch + 1 >= min(ocfg.epochs, 200):
            log("[official-rqvae] tiny dataset early stop for practicality")
            break

    if out_path is not None and (out_path / "best_collision_model.pth").exists():
        ckpt = torch.load(
            out_path / "best_collision_model.pth", map_location=dev, weights_only=False
        )
        model.load_state_dict(ckpt["state_dict"])
        log(
            f"[official-rqvae] loaded best_collision_model "
            f"(epoch={ckpt['epoch']} collision={ckpt['metric']:.4f})"
        )

    return model, encode_all_minionerec(model, all_x, use_sk=False)


def resolve_collisions_minionerec(
    model: MiniOneRecRQVAE,
    features: np.ndarray | torch.Tensor,
    codes: np.ndarray,
    *,
    sk_epsilon: float = 0.003,
    max_iters: int = 20,
    device: str = "cuda:0",
    log: Any = print,
) -> np.ndarray:
    """Port of MiniOneRec ``generate_indices.py`` collision loop.

    Official comment: duplicate items in the dataset are accepted; zero collision
    is not an unconditional hard requirement.
    """
    from llm4rec.sid.base import collision_groups, collision_rate

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    x = (
        features
        if isinstance(features, torch.Tensor)
        else torch.tensor(features, dtype=torch.float32)
    ).to(dev)

    # Official: disable Sinkhorn on all but last VQ layer
    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = float(sk_epsilon)

    out = codes.copy()
    it = 0
    while True:
        rate = collision_rate(out)
        if it >= max_iters or rate == 0.0:
            log(f"[sid] official Sinkhorn resolution: iters={it} collision={rate:.4f}")
            break
        groups = collision_groups(out)
        log(f"[sid]   round {it + 1}: {len(groups)} collision groups")
        for members in groups:
            batch = x[torch.tensor(members, device=dev)]
            idx = model.get_indices(batch, use_sk=True)
            idx_np = idx.view(-1, idx.shape[-1]).cpu().numpy()
            for pos, item_idx in enumerate(members):
                out[item_idx] = idx_np[pos]
        it += 1
    return out
