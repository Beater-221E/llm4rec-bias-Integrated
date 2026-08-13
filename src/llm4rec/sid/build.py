"""``build-sid``：生成 Semantic ID 静态产物。

``mode: reproduction`` (MiniOneRec):
  official RQ-VAE — no PCA, layers=[2048..64], e_dim=32, codebooks=[256,256,256],
  Sinkhorn collision resolution. Non-zero final collision is accepted unless
  ``sid.strict_unique=true``.

``mode: integrated``:
  generalized RQ-VAE / rqkmeans with optional PCA (existing research path).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from llm4rec.core.exceptions import ConfigurationError
from llm4rec.core.modes import get_mode
from llm4rec.data.base import get_adapter
from llm4rec.sid.artifact import (
    CODEBOOK_NAME,
    SID_MAP_NAME,
    STATS_NAME,
    artifact_dir,
    items_fingerprint,
    write_manifest,
)
from llm4rec.sid.base import compute_collision_metrics, find_duplicate_text_groups
from llm4rec.sid.collision import enforce_unique_last_code, resolve_collisions_sinkhorn
from llm4rec.sid.embeddings import encode_items
from llm4rec.sid.rqvae import apply_pca, collision_rate, train_residual_kmeans, train_rqvae
from llm4rec.sid.table import format_sid


def _use_reference_sid(cfg: dict[str, Any], sid_cfg: dict[str, Any]) -> bool:
    """MiniOneRec reference SID when reproduction mode or explicitly requested."""
    if str(sid_cfg.get("implementation") or "").lower() in {
        "official",
        "minionerec_reference",
        "minionerec",
    }:
        return True
    route = str((cfg.get("experiment") or {}).get("route") or "")
    return get_mode(cfg) == "reproduction" and route == "minionerec"


def _collision_handling_label(*, official: bool, method: str, sid_cfg: dict[str, Any]) -> str:
    explicit = sid_cfg.get("collision_handling")
    if explicit:
        return str(explicit)
    if method == "rqkmeans":
        if bool((sid_cfg.get("rqkmeans") or {}).get("enforce_unique", True)):
            return "unique_last_code"
        return "none"
    if official:
        return "minionerec_sinkhorn_last_level"
    return "sinkhorn_last_level"


def build_sid(cfg: dict[str, Any], *, force: bool = False, log: Any = print) -> Path:
    import time

    t_total0 = time.perf_counter()
    phase_sec: dict[str, float] = {}
    sid_cfg = cfg["sid"]
    data_cfg = cfg["data"]
    adapter = get_adapter(cfg)
    dataset = adapter.dataset_key(cfg)
    seed = int(cfg.get("seed") or 42)
    # RQ-VAE train seed is independent of the global run seed (MiniOneRec: 2024).
    rqvae_seed = int((sid_cfg.get("rqvae") or {}).get("seed") or (2024 if _use_reference_sid(cfg, sid_cfg) else seed))
    mode = get_mode(cfg)
    official = _use_reference_sid(cfg, sid_cfg)

    meta = adapter.load_item_meta(cfg)
    item_ids = sorted(meta.keys())
    if not item_ids:
        raise ConfigurationError("item_meta.json 是空的")
    fingerprint = items_fingerprint(item_ids)

    # Stamp implementation into sid_cfg so artifact hash distinguishes paths
    sid_cfg = dict(sid_cfg)
    sid_cfg["implementation"] = "minionerec_reference" if official else "integrated"
    sid_cfg["collision_handling"] = _collision_handling_label(
        official=official, method=str(sid_cfg.get("method") or "rqvae"), sid_cfg=sid_cfg
    )
    cfg = {**cfg, "sid": sid_cfg}

    out_dir = artifact_dir(
        sid_cfg,
        dataset=dataset,
        seed=seed,
        items_fp=fingerprint,
        root=Path(cfg["paths"].get("artifacts_dir") or "artifacts") / "sid",
    )
    if (out_dir / SID_MAP_NAME).is_file() and not force:
        log(f"[sid] 已存在，跳过：{out_dir}")
        return out_dir

    texts = [str(meta[i].get("text") or meta[i].get("title") or "") for i in item_ids]
    t0 = time.perf_counter()
    emb, _ = encode_items(cfg, item_ids, texts, force=force, log=log)
    phase_sec["text_embedding_sec"] = round(time.perf_counter() - t0, 4)
    if emb.shape[0] != len(item_ids):
        raise ConfigurationError(
            f"embedding 行数 {emb.shape[0]} 与物品数 {len(item_ids)} 对不上"
        )

    levels = int(sid_cfg.get("levels") or 3)
    codebook_size = int(sid_cfg.get("codebook_size") or 256)
    method = str(sid_cfg.get("method") or "rqvae").lower()
    device = str((sid_cfg.get("encoder") or {}).get("device") or "cuda:0")
    dup_groups = find_duplicate_text_groups(texts)

    model = None
    pca_stats: dict[str, Any] | None = None
    features = emb.astype(np.float32)

    if official:
        if method != "rqvae":
            raise ConfigurationError(
                "MiniOneRec reproduction SID 必须使用 method=rqvae "
                f"（得到 method={method}）"
            )
        from llm4rec.sid.minionerec_rqvae import (
            MiniOneRecRQVAEConfig,
            resolve_collisions_minionerec,
            train_minionerec_rqvae,
        )

        rq_cfg = dict(sid_cfg.get("rqvae") or {})
        # Reproduction defaults: never PCA; official architecture
        rq_cfg["pca_dim"] = None
        if "num_emb_list" not in rq_cfg:
            rq_cfg["num_emb_list"] = [codebook_size] * levels
        ocfg = MiniOneRecRQVAEConfig.from_dict(rq_cfg)
        if mode == "reproduction":
            # Keep codebook at configured size but warn if not official 256
            if ocfg.num_emb_list != [256, 256, 256]:
                log(
                    f"[sid] WARNING: reproduction usually uses [256,256,256], "
                    f"got {ocfg.num_emb_list} (experimental variant)"
                )
        log(
            f"[sid] MiniOneRec official RQ-VAE (mode={mode}): "
            f"NO PCA, layers={ocfg.layers}, e_dim={ocfg.e_dim}, "
            f"codebooks={ocfg.num_emb_list}"
        )
        t_rq0 = time.perf_counter()
        model, codes = train_minionerec_rqvae(
            features,
            ocfg,
            seed=rqvae_seed,
            device=device,
            out_dir=out_dir,
            log=log,
        )
        phase_sec["rqvae_train_sec"] = round(time.perf_counter() - t_rq0, 4)
        levels = len(ocfg.num_emb_list)
        codebook_size = int(ocfg.num_emb_list[0])
    else:
        # Integrated path: optional PCA + simplified / rqkmeans
        rq_block = sid_cfg.get(method) or sid_cfg.get("rqvae") or {}
        pca_dim = rq_block.get("pca_dim")
        if pca_dim not in (None, 0, False):
            features, pca_stats = apply_pca(emb, int(pca_dim), seed=seed)
            log(
                f"[sid] PCA {pca_stats['original_dim']} → {pca_stats['pca_dim']}"
                f"（保留方差 {pca_stats['explained_variance_ratio_sum']:.4f}）"
            )
        else:
            log("[sid] integrated: PCA disabled")

        t_rq0 = time.perf_counter()
        if method == "rqvae":
            model, codes = train_rqvae(
                features,
                sid_cfg.get("rqvae") or {},
                levels=levels,
                codebook_size=codebook_size,
                seed=int((sid_cfg.get("rqvae") or {}).get("seed") or seed),
                device=device,
                out_dir=out_dir,
                log=log,
            )
        elif method == "rqkmeans":
            codes = train_residual_kmeans(
                features, levels=levels, codebook_size=codebook_size, seed=seed, log=log
            )
            if bool((sid_cfg.get("rqkmeans") or {}).get("enforce_unique", True)):
                codes = enforce_unique_last_code(codes, codebook_size)
        else:
            raise ConfigurationError(f"未知 sid.method '{method}'（可用：rqvae / rqkmeans）")
        phase_sec["rqvae_train_sec"] = round(time.perf_counter() - t_rq0, 4)

    phase_sec["encoding_sec"] = phase_sec.get("rqvae_train_sec", 0.0)
    raw_codes = codes.copy()
    raw_collision = collision_rate(codes)
    log(f"[sid] 量化完成，原始碰撞率 = {raw_collision:.4f}")

    # Official / integrated Sinkhorn collision resolution (rqvae only)
    t_col0 = time.perf_counter()
    if method == "rqvae" and model is not None:
        import torch

        sk_eps = float(
            (sid_cfg.get("rqvae") or {}).get("sk_epsilon_last")
            or (sid_cfg.get("rqvae") or {}).get("sk_epsilon")
            or 0.003
        )
        max_iters = int((sid_cfg.get("rqvae") or {}).get("collision_retry_iters") or 20)
        if official:
            from llm4rec.sid.minionerec_rqvae import resolve_collisions_minionerec

            codes = resolve_collisions_minionerec(
                model,
                features,
                codes,
                sk_epsilon=sk_eps,
                max_iters=max_iters,
                device=device,
                log=log,
            )
        else:
            codes = resolve_collisions_sinkhorn(
                model,
                torch.tensor(features, dtype=torch.float32),
                codes,
                sk_epsilon=sk_eps,
                max_iters=max_iters,
                device=device,
                log=log,
            )
    phase_sec["collision_resolution_sec"] = round(time.perf_counter() - t_col0, 4)

    metrics = compute_collision_metrics(
        codes, raw_codes=raw_codes, duplicate_groups=dup_groups
    )
    log(
        f"[sid] collision metrics: "
        f"raw={metrics.raw_collision_rate:.4f} "
        f"post={metrics.post_resolution_collision_rate:.4f} "
        f"groups={metrics.num_collision_groups} "
        f"max_group={metrics.max_collision_group_size} "
        f"dup={metrics.duplicate_item_collision_rate:.4f} "
        f"quant={metrics.quantization_collision_rate:.4f}"
    )

    strict_unique = bool(sid_cfg.get("strict_unique", False))
    # Never treat max_collision_rate==0 as an implicit hard fail in reproduction.
    max_rate = sid_cfg.get("max_collision_rate")
    if strict_unique and metrics.n_unique_sids != metrics.n_items:
        raise ConfigurationError(
            f"sid.strict_unique=true 但仍有碰撞：{metrics.n_items} items → "
            f"{metrics.n_unique_sids} unique SIDs "
            f"(post_resolution_collision_rate={metrics.post_resolution_collision_rate:.4f})"
        )
    if max_rate is not None and float(max_rate) < 1.0:
        if metrics.post_resolution_collision_rate > float(max_rate) + 1e-12:
            if mode == "reproduction" and float(max_rate) == 0.0:
                log(
                    f"[sid] WARNING: post-resolution collision "
                    f"{metrics.post_resolution_collision_rate:.4f} > 0；"
                    f"官方流程接受非零碰撞（重复商品等）。"
                    f"设置 sid.strict_unique=true 才会硬失败。"
                )
            elif float(max_rate) > 0.0:
                raise ConfigurationError(
                    f"SID 碰撞率 {metrics.post_resolution_collision_rate:.4f} "
                    f"超过 sid.max_collision_rate={max_rate}"
                )

    prefixes = list(sid_cfg.get("layer_prefixes") or ["a", "b", "c"])[:levels]
    if len(prefixes) < levels:
        raise ConfigurationError(f"sid.layer_prefixes 长度不足 {levels}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collision-safe map: each item keeps its codes; reverse collisions preserved in table
    sid_map = {
        item: {
            "codes": [int(c) for c in row],
            "sid": format_sid([int(c) for c in row], prefixes),
        }
        for item, row in zip(item_ids, codes, strict=True)
    }
    (out_dir / SID_MAP_NAME).write_text(
        json.dumps(sid_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if model is not None:
        import torch

        payload: dict[str, Any]
        if official:
            payload = {
                "implementation": "minionerec_reference",
                "state_dict": model.state_dict(),
                "codebooks": [vq.get_codebook().detach().cpu() for vq in model.rq.vq_layers],
            }
        else:
            payload = {
                "implementation": "integrated_simplified",
                "codebooks": [cb.detach().cpu() for cb in model.codebooks],
                "encoder": model.encoder.state_dict(),
                "decoder": model.decoder.state_dict(),
            }
        torch.save(payload, out_dir / CODEBOOK_NAME)

    usage = {
        f"layer_{i}": int(len(set(int(c) for c in codes[:, i])))
        for i in range(codes.shape[1])
    }
    (out_dir / STATS_NAME).write_text(
        json.dumps(
            {
                "mode": mode,
                "implementation": sid_cfg["implementation"],
                "method": method,
                "n_items": len(item_ids),
                "levels": levels,
                "codebook_size": codebook_size,
                "pca": pca_stats,
                "codebook_usage": usage,
                **metrics.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    sid_cfg_out = dict(sid_cfg)
    sid_cfg_out["levels"] = levels
    sid_cfg_out["codebook_size"] = codebook_size
    sid_cfg_out["layer_prefixes"] = prefixes
    write_manifest(
        out_dir,
        sid_cfg=sid_cfg_out,
        dataset=dataset,
        seed=seed,
        items_fp=fingerprint,
        n_items=len(item_ids),
        collision_rate=metrics.post_resolution_collision_rate,
    )

    log(f"[sid] 完成 → {out_dir}")
    log(
        f"[sid]   物品 {len(item_ids)}  层数 {levels}  "
        f"post_collision={metrics.post_resolution_collision_rate:.4f}"
    )
    phase_sec["total_sec"] = round(time.perf_counter() - t_total0, 4)
    phase_sec["items_per_sec"] = round(
        len(item_ids) / max(phase_sec["total_sec"], 1e-6), 3
    )
    cfg.setdefault("_performance", {})["sid"] = dict(phase_sec)
    log(f"[sid] performance={phase_sec}")
    return out_dir


def resolve_for_training(cfg: dict[str, Any]) -> Path:
    from llm4rec.sid.artifact import resolve_sid_dir

    adapter = get_adapter(cfg)
    meta = adapter.load_item_meta(cfg)
    sid_cfg = dict(cfg["sid"])
    if _use_reference_sid(cfg, sid_cfg):
        sid_cfg.setdefault("implementation", "minionerec_reference")
        sid_cfg.setdefault("collision_handling", "minionerec_sinkhorn_last_level")
    else:
        sid_cfg.setdefault("implementation", "integrated")
        sid_cfg.setdefault(
            "collision_handling",
            _collision_handling_label(
                official=False, method=str(sid_cfg.get("method") or "rqvae"), sid_cfg=sid_cfg
            ),
        )
    return resolve_sid_dir(
        sid_cfg,
        dataset=adapter.dataset_key(cfg),
        seed=int(cfg.get("seed") or 42),
        item_ids=sorted(meta.keys()),
        root=Path(cfg["paths"].get("artifacts_dir") or "artifacts") / "sid",
        strict=True,
    )
