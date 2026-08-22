"""Semantic-ID transition teacher (SDA).

Mapped from ``dragonfly90/llm4rec-bias`` ``src/llm4rec/sid_transition.py``:

* ``TransitionModel`` → same class name, variable levels / per-level K
* ``build_windows`` → ``windows_from_examples`` (uses ``history`` / ``target_item``)
* ``Transition`` → ``SidTransitionTeacher`` (GPU-batched catalog scoring)
* ``main`` → ``run_transition`` (Integrated checkpoint + fingerprint)

Not copied: MovieLens loaders, fixed ``levels=3`` / ``K=64``, MPS device
selection, dropping the last SID layer by default.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm4rec.core import distributed as dist_utils
from llm4rec.core.exceptions import CheckpointError, ConfigurationError

PAD = -1
CHECKPOINT_NAME = "transition.pt"
MANIFEST_NAME = "transition_manifest.json"

# Manifest keys that *explicitly* mark the last SID layer as collision-only.
# ``collision_handling: sinkhorn_last_level`` is a *build* algorithm, not this.
_COLLISION_ONLY_KEYS = (
    "last_level_collision_only",
    "last_level_is_collision",
    "drop_collision_level",
)


def semantic_level_count(sid_table: Any, sid_cfg: dict[str, Any] | None = None) -> int:
    """How many SID levels the MLP models.

    Defaults to every layer. The last layer is dropped only when the SID
    manifest / config *explicitly* marks it as collision-resolution-only.
    """
    levels = int(sid_table.levels)
    merged: dict[str, Any] = {}
    manifest_cfg = getattr(getattr(sid_table, "manifest", None), "sid_config", None)
    if isinstance(manifest_cfg, dict):
        merged.update(manifest_cfg)
    if sid_cfg:
        merged.update(sid_cfg)

    if any(bool(merged.get(k)) for k in _COLLISION_ONLY_KEYS):
        return max(1, levels - 1)
    collision_level = merged.get("collision_level")
    if collision_level == "last":
        return max(1, levels - 1)
    if isinstance(collision_level, int) and collision_level == levels - 1:
        return max(1, levels - 1)
    return levels


def windows_from_examples(
    examples: Sequence[dict[str, Any]],
    sid_table: Any,
    history_max_length: int,
) -> list[tuple[list[str], str]]:
    """(history, target) pairs from Integrated examples.

    Replaces reference ``build_windows``: no raw MovieLens sequences, no
    ``holdout=2`` (train/val are already split).
    """
    hist_len = max(1, int(history_max_length))
    pairs: list[tuple[list[str], str]] = []
    for row in examples:
        target = str(row.get("target_item") or "")
        if not target or target not in sid_table:
            continue
        history = [
            str(item)
            for item in (row.get("history") or [])
            if str(item) in sid_table
        ]
        if not history:
            continue
        pairs.append((history[-hist_len:], target))
    return pairs


def encode_histories(
    histories: Sequence[Sequence[str]],
    sid_table: Any,
    *,
    levels: int,
    history_max_length: int,
    device: torch.device | str,
) -> torch.Tensor:
    """list[list[item_id]] → ``(B, T, L)`` PAD-filled code tensor."""
    batch = len(histories)
    hist_len = max(1, int(history_max_length))
    out = torch.full((batch, hist_len, levels), PAD, dtype=torch.long)
    for b, history in enumerate(histories):
        items = [str(item) for item in history if str(item) in sid_table][-hist_len:]
        for t, item in enumerate(items):
            codes = sid_table.codes[str(item)][:levels]
            out[b, t, : len(codes)] = torch.tensor(codes, dtype=torch.long)
    return out.to(device)


# ------------------------------------------------------------------ model


class TransitionModel(nn.Module):
    """History of SID codes → hierarchical next-SID distribution.

    Encoder (from the reference ``TransitionModel``):

    * per-level code embeddings
    * recency-weighted mean pooling of the history
    * last-item representation
    * per-level SID frequency / histogram

    Heads: one softmax per level, conditioned on codes already chosen above —

    ``P(s_1, …, s_L | h) = ∏_l P(s_l | h, s_<l)``
    """

    def __init__(
        self,
        codebook_sizes: Sequence[int],
        *,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        decay: float = 0.9,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        sizes = [int(k) for k in codebook_sizes]
        if not sizes or any(k < 1 for k in sizes):
            raise ConfigurationError(f"invalid codebook_sizes={list(codebook_sizes)}")
        self.codebook_sizes = sizes
        self.levels = len(sizes)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.decay = float(decay)

        self.code_emb = nn.ModuleList(
            [nn.Embedding(k, self.embedding_dim) for k in sizes]
        )
        hist_dim = sum(sizes)
        self.enc = nn.Sequential(
            nn.Linear(2 * self.embedding_dim + hist_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim + layer * self.embedding_dim, sizes[layer])
                for layer in range(self.levels)
            ]
        )

    def encode(self, codes: torch.Tensor) -> torch.Tensor:
        """``codes``: ``(B, T, L)`` long, PAD-filled → ``(B, hidden)``."""
        mask = (codes[:, :, 0] != PAD).float()
        safe = codes.clamp(min=0)
        item = sum(
            self.code_emb[layer](safe[:, :, layer].clamp(max=self.codebook_sizes[layer] - 1))
            for layer in range(self.levels)
        )
        pos = mask.cumsum(1)
        n = mask.sum(1, keepdim=True).clamp(min=1)
        weights = (self.decay ** (n - pos)) * mask
        denom = weights.sum(1, keepdim=True).clamp(min=1e-6)
        pooled = (item * weights.unsqueeze(-1)).sum(1) / denom
        last_idx = (n.long() - 1).clamp(min=0)
        last = item.gather(
            1, last_idx.unsqueeze(-1).expand(-1, 1, item.size(-1))
        ).squeeze(1)
        hists = []
        hist_denom = mask.sum(1, keepdim=True).clamp(min=1e-6)
        for layer in range(self.levels):
            k = self.codebook_sizes[layer]
            oh = F.one_hot(safe[:, :, layer].clamp(max=k - 1), k).float()
            oh = oh * mask.unsqueeze(-1)
            hists.append(oh.sum(1) / hist_denom)
        return self.enc(torch.cat([pooled, last, *hists], dim=-1))

    def level_logits(self, hidden: torch.Tensor, prev_codes: Sequence[torch.Tensor]) -> torch.Tensor:
        """Logits for level ``len(prev_codes)``, conditioned on codes above it."""
        layer = len(prev_codes)
        ctx = [hidden]
        for j in range(layer):
            idx = prev_codes[j].clamp(min=0, max=self.codebook_sizes[j] - 1)
            ctx.append(self.code_emb[j](idx))
        return self.heads[layer](torch.cat(ctx, dim=-1))

    def joint_nll(self, codes: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-row joint NLL ``-∑_l log P(s_l* | h, s_<l*)``. Shape ``(B,)``."""
        hidden = self.encode(codes)
        nll = torch.zeros(target.size(0), device=target.device, dtype=torch.float32)
        for layer in range(self.levels):
            logits = self.level_logits(hidden, [target[:, j] for j in range(layer)])
            nll = nll + F.cross_entropy(logits, target[:, layer], reduction="none")
        return nll

    def forward(
        self,
        codes: torch.Tensor,
        target: torch.Tensor,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """Mean joint NLL (optional label smoothing). Same reduction as reference."""
        hidden = self.encode(codes)
        loss = codes.new_zeros(())
        for layer in range(self.levels):
            logits = self.level_logits(hidden, [target[:, j] for j in range(layer)])
            loss = loss + F.cross_entropy(
                logits, target[:, layer], label_smoothing=float(label_smoothing)
            )
        return loss


# ------------------------------------------------------------------ teacher


def _catalog_item_ids(sid_table: Any, catalog: Any | None) -> list[str]:
    if catalog is not None and getattr(catalog, "item_ids", None):
        return [str(i) for i in catalog.item_ids if str(i) in sid_table]
    return sorted(str(i) for i in sid_table.codes)


def _item_counts(item_ids: Sequence[str], catalog: Any | None) -> torch.Tensor:
    counts = getattr(catalog, "counts", None) or {}
    return torch.tensor(
        [float(max(int(counts.get(i, 0)), 0)) for i in item_ids],
        dtype=torch.float32,
    )


class SidTransitionTeacher:
    """Loaded ``T_φ`` + catalog bookkeeping. GPU-batched catalog scoring.

    Mapped from reference ``Transition``. Adds chunked ``log_p_all`` (no
    per-item Python loop) and Integrated string item IDs.
    """

    def __init__(
        self,
        model: TransitionModel,
        sid_table: Any,
        *,
        catalog: Any | None = None,
        history_max_length: int = 50,
        temperature: float = 1.0,
        target_smoothing: float = 0.0,
        popularity_gamma: float = 0.0,
        popularity_eps: float = 1.0,
        device: torch.device | str = "cpu",
        items: Sequence[str] | None = None,
    ) -> None:
        self.model = model
        self.sid_table = sid_table
        self.history_max_length = int(history_max_length)
        self.temperature = max(float(temperature), 1e-8)
        self.target_smoothing = float(target_smoothing)
        self.popularity_gamma = float(popularity_gamma)
        self.popularity_eps = float(popularity_eps)
        self.device = torch.device(device)
        self.levels = int(model.levels)

        self.items = list(items) if items is not None else _catalog_item_ids(sid_table, catalog)
        if not self.items:
            raise ConfigurationError("Transition teacher catalog is empty")
        self.item_pos = {item: idx for idx, item in enumerate(self.items)}

        groups: dict[tuple[int, ...], list[str]] = defaultdict(list)
        codes_rows: list[list[int]] = []
        for item in self.items:
            codes = tuple(int(c) for c in sid_table.codes[str(item)][: self.levels])
            groups[codes].append(item)
            codes_rows.append(list(codes))
        self.group = groups
        self.item_codes = torch.tensor(codes_rows, dtype=torch.long, device=self.device)
        self.log_group_size = torch.tensor(
            [
                math.log(len(groups[tuple(sid_table.codes[item][: self.levels])]))
                for item in self.items
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.item_level1 = self.item_codes[:, 0].contiguous()
        counts = _item_counts(self.items, catalog).to(self.device)
        self.log_pop = -self.popularity_gamma * torch.log(counts + self.popularity_eps)
        self.model.to(self.device).eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        sid_table: Any,
        *,
        catalog: Any | None = None,
        device: torch.device | str = "cpu",
        temperature: float | None = None,
        target_smoothing: float | None = None,
        popularity_gamma: float | None = None,
    ) -> SidTransitionTeacher:
        blob = load_transition_checkpoint(path, sid_table)
        cfg = blob["config"]
        model = TransitionModel(
            codebook_sizes=cfg["codebook_sizes"],
            embedding_dim=int(cfg.get("embedding_dim") or cfg.get("d") or 128),
            hidden_dim=int(cfg.get("hidden_dim") or cfg.get("hidden") or 256),
            decay=float(cfg.get("decay") or 0.9),
            dropout=0.0,
        )
        model.load_state_dict(blob["state_dict"])
        return cls(
            model,
            sid_table,
            catalog=catalog,
            history_max_length=int(cfg.get("history_max_length") or cfg.get("history_len") or 50),
            temperature=float(cfg["temperature"] if temperature is None else temperature),
            target_smoothing=float(
                cfg["target_smoothing"] if target_smoothing is None else target_smoothing
            ),
            popularity_gamma=float(
                cfg["popularity_gamma"] if popularity_gamma is None else popularity_gamma
            ),
            device=device,
        )

    def _encode(self, histories: Sequence[Sequence[str]]) -> torch.Tensor:
        codes = encode_histories(
            histories,
            self.sid_table,
            levels=self.levels,
            history_max_length=self.history_max_length,
            device=self.device,
        )
        return self.model.encode(codes)

    def _joint_logp_codes(self, hidden: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """``hidden`` ``(B, H)``, ``codes`` ``(B, L)`` → joint log P of each row."""
        logp = torch.zeros(hidden.size(0), device=hidden.device, dtype=torch.float32)
        for layer in range(self.levels):
            logits = self.model.level_logits(hidden, [codes[:, j] for j in range(layer)])
            step = F.log_softmax(logits, dim=-1).gather(1, codes[:, layer : layer + 1])
            logp = logp + step.squeeze(1)
        return logp

    def _calibrate(self, logp: torch.Tensor) -> torch.Tensor:
        """``P*(i|h) ∝ P_T(i|h)^{1/τ} · (count(i)+ε)^{-γ}``, then uniform mix."""
        calibrated = logp / self.temperature + self.log_pop.unsqueeze(0)
        calibrated = F.log_softmax(calibrated, dim=-1)
        mix = self.target_smoothing
        if mix <= 0:
            return calibrated
        if mix >= 1:
            return torch.full_like(calibrated, -math.log(calibrated.size(-1)))
        floor = math.log(mix / calibrated.size(-1))
        return torch.logaddexp(
            calibrated + math.log1p(-mix),
            calibrated.new_full((), floor),
        )

    @torch.no_grad()
    def log_p_all(
        self,
        histories: Sequence[Sequence[str]],
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """``(B, |catalog|)`` calibrated log P*(item | history). GPU + chunked."""
        hidden = self._encode(histories)
        batch, n_items = hidden.size(0), len(self.items)
        chunk = max(1, int(chunk_size))
        parts: list[torch.Tensor] = []
        for start in range(0, n_items, chunk):
            end = min(n_items, start + chunk)
            codes = self.item_codes[start:end]
            width = codes.size(0)
            flat_h = hidden.unsqueeze(1).expand(batch, width, -1).reshape(batch * width, -1)
            flat_c = codes.unsqueeze(0).expand(batch, width, -1).reshape(batch * width, -1)
            lp = self._joint_logp_codes(flat_h, flat_c).view(batch, width)
            parts.append(lp - self.log_group_size[start:end])
        return self._calibrate(torch.cat(parts, dim=1))

    @torch.no_grad()
    def log_p_items(
        self,
        histories: Sequence[Sequence[str]],
        candidate_items: Sequence[Any],
        *,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """Log P* of candidates.

        * ``candidate_items`` length-B item ids → ``(B,)``
        * ``candidate_items`` as list-of-lists → ``(B, C)``
        """
        if not candidate_items:
            return torch.zeros(len(histories), device=self.device)
        first = candidate_items[0]
        if isinstance(first, (list, tuple)):
            full = self.log_p_all(histories, chunk_size=chunk_size)
            rows = []
            for b, cands in enumerate(candidate_items):
                idx = [self.item_pos[str(item)] for item in cands]
                rows.append(full[b, idx])
            return torch.stack(rows, dim=0)
        full = self.log_p_all(histories, chunk_size=chunk_size)
        idx = torch.tensor(
            [self.item_pos[str(item)] for item in candidate_items],
            dtype=torch.long,
            device=self.device,
        )
        return full[torch.arange(full.size(0), device=self.device), idx]

    @torch.no_grad()
    def log_p_target(
        self,
        histories: Sequence[Sequence[str]],
        target_items: Sequence[str],
        *,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """``(B,)`` calibrated log P* of each target item."""
        return self.log_p_items(histories, target_items, chunk_size=chunk_size)

    @torch.no_grad()
    def level1_probs(self, histories: Sequence[Sequence[str]]) -> torch.Tensor:
        """``P_T(C1 | h)`` from the first MLP head. Shape ``(B, K1)``."""
        hidden = self._encode(histories)
        return F.softmax(self.model.level_logits(hidden, []), dim=-1)

    @torch.no_grad()
    def level1_from_logp(self, log_p_catalog: torch.Tensor) -> torch.Tensor:
        """Catalog-marginal ``P*(C1)`` from a ``(B, N)`` item distribution."""
        probs = torch.exp(log_p_catalog - log_p_catalog.max(dim=-1, keepdim=True).values)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        k1 = int(self.model.codebook_sizes[0])
        out = torch.zeros(probs.size(0), k1, device=probs.device, dtype=probs.dtype)
        idx = self.item_level1.unsqueeze(0).expand(probs.size(0), -1)
        out.scatter_add_(1, idx, probs)
        return out

    @torch.no_grad()
    def sample_items(
        self,
        histories: Sequence[Sequence[str]],
        n_samples: int,
        *,
        generator: torch.Generator | None = None,
        chunk_size: int = 256,
    ) -> tuple[list[list[str]], torch.Tensor, torch.Tensor]:
        """Sample ``n_samples`` catalog items *from* P* (with replacement).

        Returns ``(sampled_ids, log_p_all, level1_catalog_probs)``.
        """
        logp = self.log_p_all(histories, chunk_size=chunk_size)
        probs = torch.exp(logp - logp.max(dim=-1, keepdim=True).values)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        # multinomial + Generator is reliable on CPU
        draw = torch.multinomial(
            probs.cpu(),
            num_samples=max(1, int(n_samples)),
            replacement=True,
            generator=generator,
        )
        sampled = [[self.items[int(j)] for j in row.tolist()] for row in draw]
        return sampled, logp, self.level1_from_logp(logp)


# ------------------------------------------------------------------ checkpoint I/O


def resolve_transition_path(path: str | Path) -> Path:
    root = Path(path)
    if root.is_file():
        return root
    for cand in (root / CHECKPOINT_NAME, root / "final" / CHECKPOINT_NAME):
        if cand.is_file():
            return cand
    raise CheckpointError(f"找不到 Transition checkpoint：{path}")


def _assert_sid_fingerprint(blob: dict[str, Any], sid_table: Any) -> None:
    saved = blob.get("sid_fingerprint") or {}
    if not saved:
        raise CheckpointError("Transition checkpoint 缺少 sid_fingerprint")
    current = sid_table.fingerprint()
    for key in ("config_hash", "items_fingerprint", "levels", "codebook_sizes"):
        if key not in saved:
            continue
        if saved[key] != current[key]:
            raise CheckpointError(
                f"Transition checkpoint 与当前 SID 表不一致：{key} "
                f"saved={saved[key]} current={current[key]}"
            )


def load_transition_checkpoint(path: str | Path, sid_table: Any) -> dict[str, Any]:
    ckpt = resolve_transition_path(path)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "state_dict" not in blob or "config" not in blob:
        raise CheckpointError(f"非法 Transition checkpoint：{ckpt}")
    _assert_sid_fingerprint(blob, sid_table)
    return blob


def save_transition_checkpoint(
    output_dir: Path,
    *,
    model: TransitionModel,
    config: dict[str, Any],
    sid_fingerprint: dict[str, Any],
    data: dict[str, Any],
    metrics: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": dict(config),
        "sid_fingerprint": dict(sid_fingerprint),
        "data": dict(data),
        "metrics": dict(metrics),
    }
    ckpt = output_dir / CHECKPOINT_NAME
    torch.save(payload, ckpt)
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "checkpoint": ckpt.name,
                "sid_fingerprint": sid_fingerprint,
                "config": config,
                "data": data,
                "metrics": metrics,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return ckpt


# ------------------------------------------------------------------ training


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{dist_utils.local_rank()}")
    return torch.device("cpu")


@torch.no_grad()
def _mean_joint_nll(
    model: TransitionModel,
    codes: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    bs = max(1, int(batch_size))
    for start in range(0, codes.size(0), bs):
        end = min(codes.size(0), start + bs)
        nll = model.joint_nll(codes[start:end], target[start:end])
        total += float(nll.sum().item())
        count += int(nll.numel())
    return total / max(count, 1)


def run_transition(
    *,
    cfg: dict[str, Any],
    sid_table: Any,
    catalog: Any | None,
    train_examples: Sequence[dict[str, Any]],
    val_examples: Sequence[dict[str, Any]],
    output_dir: Path,
    logger: Any,
) -> dict[str, Any]:
    """Train the Transition MLP on the train split; select by val joint NLL."""
    tcfg = ((cfg.get("train") or {}).get("transition") or {})
    out = Path(output_dir)
    ckpt_path = out / CHECKPOINT_NAME
    if not dist_utils.is_main():
        dist_utils.barrier("transition_done")
        return {
            "stage": "transition",
            "checkpoint": str(ckpt_path),
            "n_train": 0,
            "n_val": 0,
        }

    history_len = int(tcfg.get("history_max_length") or 50)
    sid_cfg = dict(cfg.get("sid") or {})
    levels = semantic_level_count(sid_table, sid_cfg)
    codebook_sizes = list(sid_table.level_codebook_sizes()[:levels])
    train_pairs = windows_from_examples(train_examples, sid_table, history_len)
    val_pairs = windows_from_examples(val_examples, sid_table, history_len)
    if not train_pairs:
        raise ConfigurationError("transition 训练集为空（需要 history + target_item）")

    device = _device()
    train_h = encode_histories(
        [h for h, _ in train_pairs],
        sid_table,
        levels=levels,
        history_max_length=history_len,
        device=device,
    )
    train_t = torch.tensor(
        [list(sid_table.codes[t][:levels]) for _, t in train_pairs],
        dtype=torch.long,
        device=device,
    )
    val_h = (
        encode_histories(
            [h for h, _ in val_pairs],
            sid_table,
            levels=levels,
            history_max_length=history_len,
            device=device,
        )
        if val_pairs
        else None
    )
    val_t = (
        torch.tensor(
            [list(sid_table.codes[t][:levels]) for _, t in val_pairs],
            dtype=torch.long,
            device=device,
        )
        if val_pairs
        else None
    )

    model = TransitionModel(
        codebook_sizes,
        embedding_dim=int(tcfg.get("embedding_dim") or 128),
        hidden_dim=int(tcfg.get("hidden_dim") or 256),
        decay=float(tcfg.get("decay") or 0.9),
        dropout=float(tcfg.get("dropout") or 0.3),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg.get("learning_rate") or 1e-3),
        weight_decay=float(tcfg.get("weight_decay") or 0.0),
    )
    epochs = max(1, int(tcfg.get("epochs") or 10))
    max_steps = tcfg.get("max_steps")
    batch_size = max(1, int(tcfg.get("batch_size") or 256))
    label_smoothing = float(tcfg.get("label_smoothing") or 0.0)
    n_train = train_h.size(0)
    logger.info(
        f"[transition] train={n_train} val={len(val_pairs)} "
        f"levels={levels} codebook_sizes={codebook_sizes} device={device}"
    )

    best = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history: list[dict[str, Any]] = []
    step = 0
    stop = False
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        running = 0.0
        seen = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            loss = model(train_h[idx], train_t[idx], label_smoothing=label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach()) * int(idx.numel())
            seen += int(idx.numel())
            step += 1
            if max_steps not in (None, 0, "null") and step >= int(max_steps):
                stop = True
                break
        train_nll = running / max(seen, 1)
        val_nll = (
            _mean_joint_nll(model, val_h, val_t, batch_size)
            if val_h is not None and val_t is not None
            else train_nll
        )
        flag = ""
        if val_nll < best:
            best = val_nll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            flag = " <- best"
        record = {
            "epoch": epoch + 1,
            "step": step,
            "train_joint_nll": train_nll,
            "val_joint_nll": val_nll,
        }
        history.append(record)
        logger.info(
            f"[transition] epoch {epoch + 1}/{epochs} "
            f"train joint NLL {train_nll:.4f} val joint NLL {val_nll:.4f}{flag}"
        )
        logger.log_metrics(
            {"train_joint_nll": train_nll, "val_joint_nll": val_nll},
            stage="transition",
            step=step,
            epoch=float(epoch + 1),
            split="val",
            wandb_prefix="transition",
        )
        if stop:
            break

    model.load_state_dict(best_state)
    config = {
        "codebook_sizes": codebook_sizes,
        "levels": levels,
        "embedding_dim": int(tcfg.get("embedding_dim") or 128),
        "hidden_dim": int(tcfg.get("hidden_dim") or 256),
        "decay": float(tcfg.get("decay") or 0.9),
        "dropout": float(tcfg.get("dropout") or 0.3),
        "history_max_length": history_len,
        "temperature": float(tcfg.get("temperature") or 1.0),
        "target_smoothing": float(tcfg.get("target_smoothing") or 0.0),
        "popularity_gamma": float(tcfg.get("popularity_gamma") or 0.0),
        "label_smoothing": label_smoothing,
        "learning_rate": float(tcfg.get("learning_rate") or 1e-3),
        "weight_decay": float(tcfg.get("weight_decay") or 0.0),
        "batch_size": batch_size,
        "epochs": epochs,
    }
    data_meta = {
        "n_train": n_train,
        "n_val": len(val_pairs),
        "history_max_length": history_len,
        "split": {"train": "train_examples", "val": "val_examples"},
    }
    metrics = {"best_val_joint_nll": best, "train_steps": step}
    path = save_transition_checkpoint(
        out,
        model=model,
        config=config,
        sid_fingerprint=sid_table.fingerprint(),
        data=data_meta,
        metrics=metrics,
    )
    (out / "train_log.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"[transition] 完成（val joint NLL {best:.4f}）→ {path}")
    dist_utils.barrier("transition_done")
    return {
        "stage": "transition",
        "checkpoint": str(path),
        "metrics": metrics,
        "n_train": n_train,
        "n_val": len(val_pairs),
        "config": config,
        "sid_fingerprint": sid_table.fingerprint(),
    }
