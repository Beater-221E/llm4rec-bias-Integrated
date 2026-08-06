"""``build-sid``：生成 Semantic ID 静态产物。

只在 ``prepare.sh`` 里跑一次。训练和评测全程只读这份产物。

流程：
    item_meta.json → 物品文本 → (已有 embedding) → PCA → RQ-VAE / RQ-Kmeans
    → 碰撞消解 → 校验唯一性 → 写 artifacts/sid/<dataset>/<hash>/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from llm4rec.core.exceptions import ConfigurationError
from llm4rec.data.base import get_adapter
from llm4rec.sid.artifact import (
    CODEBOOK_NAME,
    SID_MAP_NAME,
    STATS_NAME,
    artifact_dir,
    items_fingerprint,
    write_manifest,
)
from llm4rec.sid.embeddings import encode_items
from llm4rec.sid.rqvae import (
    apply_pca,
    break_collisions_extra_level,
    collision_rate,
    enforce_unique_last_code,
    train_residual_kmeans,
    train_rqvae,
)
from llm4rec.sid.table import DEFAULT_PREFIXES, format_sid


def build_sid(cfg: dict[str, Any], *, force: bool = False, log: Any = print) -> Path:
    sid_cfg = cfg["sid"]
    data_cfg = cfg["data"]
    adapter = get_adapter(cfg)
    dataset = adapter.dataset_key(cfg)
    seed = int(cfg.get("seed") or 42)

    meta = adapter.load_item_meta(cfg)
    item_ids = sorted(meta.keys())
    if not item_ids:
        raise ConfigurationError("item_meta.json 是空的")
    fingerprint = items_fingerprint(item_ids)

    out_dir = artifact_dir(
        sid_cfg,
        dataset=dataset,
        seed=seed,
        items_fp=fingerprint,
        root=Path(cfg["paths"].get("artifacts_dir") or "artifacts") / "sid",
    )
    if (out_dir / SID_MAP_NAME).is_file() and not force:
        log(f"[sid] 已存在，跳过：{out_dir}")
        log("[sid] （改配置会自动换到新的 hash 目录；要重建同一份用 FORCE=1）")
        return out_dir

    # 1) 物品文本 → embedding（有缓存就复用）
    texts = [str(meta[i].get("text") or meta[i].get("title") or "") for i in item_ids]
    emb, _ = encode_items(cfg, item_ids, texts, force=force, log=log)
    if emb.shape[0] != len(item_ids):
        raise ConfigurationError(
            f"embedding 行数 {emb.shape[0]} 与物品数 {len(item_ids)} 对不上"
        )

    levels = int(sid_cfg.get("levels") or 3)
    codebook_size = int(sid_cfg.get("codebook_size") or 256)
    method = str(sid_cfg.get("method") or "rqvae").lower()

    # 2) PCA
    pca_dim = int(
        (sid_cfg.get(method) or {}).get("pca_dim")
        or (sid_cfg.get("rqvae") or {}).get("pca_dim")
        or 256
    )
    features, pca_stats = apply_pca(emb, pca_dim, seed=seed)
    log(
        f"[sid] PCA {pca_stats['original_dim']} → {pca_stats['pca_dim']}"
        f"（保留方差 {pca_stats['explained_variance_ratio_sum']:.4f}）"
    )

    # 3) 量化
    model = None
    if method == "rqvae":
        model, codes = train_rqvae(
            features,
            sid_cfg.get("rqvae") or {},
            levels=levels,
            codebook_size=codebook_size,
            seed=seed,
            device=str((sid_cfg.get("encoder") or {}).get("device") or "cuda:0"),
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

    raw_collision = collision_rate(codes)
    log(f"[sid] 量化完成，原始碰撞率 = {raw_collision:.4f}")

    # 4) 碰撞消解
    handling = str(sid_cfg.get("collision_handling") or "extra_level")
    if raw_collision > 0 and handling == "extra_level":
        codes = break_collisions_extra_level(codes)
        levels = codes.shape[1]
        log(f"[sid] 加第 {levels} 位消解碰撞")

    final_collision = collision_rate(codes)
    max_allowed = float(sid_cfg.get("max_collision_rate") or 0.0)
    if final_collision > max_allowed:
        raise ConfigurationError(
            f"SID 碰撞率 {final_collision:.4f} 超过上限 {max_allowed}。\n"
            f"  带着碰撞的 SID 训练会让不同物品共享同一个 token 序列，"
            f"HR/NDCG 和 bias 指标全都失真。\n"
            f"  可选：把 sid.method 换成 rqkmeans（在 Amazon23 上碰撞明显更低），"
            f"或调大 sid.codebook_size。"
        )

    # 5) 唯一性硬校验
    unique = len({tuple(int(c) for c in row) for row in codes})
    if unique != len(item_ids):
        raise ConfigurationError(
            f"消解后仍不唯一：{len(item_ids)} 个物品 → {unique} 个 SID"
        )

    # 6) 落盘
    # extra_level 会把层数从 3 增到 4；配置里常只写 [a,b,c]，这里用默认前缀补齐。
    configured = list(sid_cfg.get("layer_prefixes") or DEFAULT_PREFIXES)
    if len(configured) < levels:
        configured = list(configured) + [
            p for p in DEFAULT_PREFIXES if p not in configured
        ]
    if len(configured) < levels:
        # 仍不够就按字母续写（极端情况）
        for i in range(len(configured), levels):
            configured.append(chr(ord("a") + i))
    prefixes = configured[:levels]
    out_dir.mkdir(parents=True, exist_ok=True)
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

        torch.save(
            {
                "codebooks": [cb.detach().cpu() for cb in model.codebooks],
                "encoder": model.encoder.state_dict(),
                "decoder": model.decoder.state_dict(),
            },
            out_dir / CODEBOOK_NAME,
        )

    usage = {
        f"layer_{i}": int(len(set(int(c) for c in codes[:, i])))
        for i in range(codes.shape[1])
    }
    (out_dir / STATS_NAME).write_text(
        json.dumps(
            {
                "method": method,
                "n_items": len(item_ids),
                "levels": levels,
                "codebook_size": codebook_size,
                "raw_collision_rate": raw_collision,
                "final_collision_rate": final_collision,
                "codebook_usage": usage,
                "pca": pca_stats,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    sid_cfg_out = dict(sid_cfg)
    sid_cfg_out["levels"] = levels
    sid_cfg_out["layer_prefixes"] = prefixes
    write_manifest(
        out_dir,
        sid_cfg=sid_cfg_out,
        dataset=dataset,
        seed=seed,
        items_fp=fingerprint,
        n_items=len(item_ids),
        collision_rate=final_collision,
    )

    log(f"[sid] 完成 → {out_dir}")
    log(f"[sid]   物品 {len(item_ids)}  层数 {levels}  碰撞率 {final_collision:.4f}")
    log(f"[sid]   逐层码本使用 {usage}")
    log("[sid] ★ 这是静态产物，训练和评测只读；训练时不会重建。")
    return out_dir


def resolve_for_training(cfg: dict[str, Any]) -> Path:
    """训练/评测时定位 SID 产物 —— 找不到就报错，绝不重建。"""
    from llm4rec.sid.artifact import resolve_sid_dir

    adapter = get_adapter(cfg)
    meta = adapter.load_item_meta(cfg)
    return resolve_sid_dir(
        cfg["sid"],
        dataset=adapter.dataset_key(cfg),
        seed=int(cfg.get("seed") or 42),
        item_ids=sorted(meta.keys()),
        root=Path(cfg["paths"].get("artifacts_dir") or "artifacts") / "sid",
        strict=True,
    )
