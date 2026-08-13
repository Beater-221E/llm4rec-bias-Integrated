"""用冻结的 text encoder 编码物品文本 → embedding。

对齐官方 MiniOneRec：把 title + description 拼成一句话，过一个冻结的
text encoder，得到的向量再交给 RQ-VAE 量化。

产物同样是**静态的**：``artifacts/embeddings/<dataset>/<encoder>/item_emb.npy``，
配 ``item_ids.json`` 保证顺序对得上。编码一次就够了 —— 换 SID 参数不用重编码。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from llm4rec.core.exceptions import MissingArtifactError

_DTYPES = {"fp16": "float16", "fp32": "float32", "bf16": "bfloat16"}


def embeddings_dir(cfg: dict[str, Any]) -> Path:
    from llm4rec.data.base import get_adapter

    encoder = str((cfg["sid"]["encoder"]).get("model") or "encoder")
    root = Path(cfg["paths"].get("artifacts_dir") or "artifacts")
    return root / "embeddings" / get_adapter(cfg).dataset_key(cfg) / encoder.replace("/", "_")


def encode_items(
    cfg: dict[str, Any],
    item_ids: list[str],
    texts: list[str],
    *,
    force: bool = False,
    log: Any = print,
) -> tuple[np.ndarray, Path]:
    """编码物品文本；已有产物且物品数一致就直接复用。"""
    import torch
    from transformers import AutoModel, AutoTokenizer

    enc_cfg = cfg["sid"]["encoder"]
    out_dir = embeddings_dir(cfg)
    emb_path = out_dir / "item_emb.npy"
    ids_path = out_dir / "item_ids.json"

    if emb_path.is_file() and ids_path.is_file() and not force:
        cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if cached_ids == item_ids:
            log(f"[embed] 复用已有 embedding：{emb_path}")
            return np.load(emb_path), emb_path
        log("[embed] 物品集合变了，重新编码")

    model_path = _resolve_encoder(enc_cfg)
    device = str(enc_cfg.get("device") or "cuda:0")
    if not torch.cuda.is_available():
        device = "cpu"
    dtype = getattr(torch, _DTYPES.get(str(enc_cfg.get("dtype") or "fp16"), "float32"))

    log(f"[embed] encoder={model_path} items={len(item_ids)} device={device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype=dtype)
    model = model.to(device).eval()

    batch_size = int(enc_cfg.get("batch_size") or 4)
    max_length = int(enc_cfg.get("max_length") or 1024)
    vectors: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            encoded = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            hidden = model(**encoded).last_hidden_state
            pooled = _mean_pool(hidden, encoded["attention_mask"])
            vectors.append(pooled.float().cpu().numpy())
            if (start // batch_size) % 200 == 0:
                log(f"[embed]   {start}/{len(texts)}")

    emb = np.concatenate(vectors, axis=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    ids_path.write_text(json.dumps(item_ids, ensure_ascii=False), encoding="utf-8")
    log(f"[embed] 完成 → {emb_path}  shape={emb.shape}")
    return emb, emb_path


def _mean_pool(hidden: Any, mask: Any) -> Any:
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)


def _resolve_encoder(enc_cfg: dict[str, Any]) -> str:
    local = enc_cfg.get("local_path")
    if local and Path(local).is_dir():
        return str(local)
    model = enc_cfg.get("model")
    if not model:
        raise MissingArtifactError("sid.encoder.model 必填")
    return str(model)


def load_embeddings(cfg: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    out_dir = embeddings_dir(cfg)
    emb_path, ids_path = out_dir / "item_emb.npy", out_dir / "item_ids.json"
    if not emb_path.is_file() or not ids_path.is_file():
        raise MissingArtifactError(
            f"缺少物品 embedding（{emb_path}），先跑 `STEPS=embed bash prepare.sh`"
        )
    return np.load(emb_path), json.loads(ids_path.read_text(encoding="utf-8"))
