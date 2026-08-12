"""Semantic ID 的**静态产物**契约。

旧版的毛病：SID 在 prepare / train 里按需构建，train 时发现缺了就偷偷重建
（``workflows/minionerec/pipeline.py`` 的 ``prepare_data``）。后果是不同 run
可能用着不同的 SID 表，而 SID 表一变，物品的语义前缀全变，
bias 指标（尤其是 exposure / coverage）根本不可比 —— 这类问题还很难发现。

新契约：

  1. SID 只由 ``prepare.sh`` 的 ``build-sid`` 步骤生成，**一次**。
  2. 产物写到 ``artifacts/sid/<dataset>/<config_hash>/``，训练与评测**只读**。
  3. 每份产物带 ``manifest.json``，记录生成它的完整配置 + 输入指纹。
  4. 训练启动时重算 hash 比对，**对不上直接抛错退出**，绝不隐式重建。

``config_hash`` 覆盖所有会改变 SID 结果的输入：数据集、编码器、层数、码本大小、
量化方法、以及物品集合本身的指纹。改任何一个都会得到一个新目录，
老产物原样保留 —— 所以已经训好的模型永远能找回它当初用的那张 SID 表。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from llm4rec.core.exceptions import ConfigurationError, MissingArtifactError

MANIFEST_NAME = "manifest.json"
SID_MAP_NAME = "item2sid.json"
CODEBOOK_NAME = "codebook.pt"
STATS_NAME = "build_stats.json"

# 只有这些配置项会影响 SID 结果，改它们才会产生新的 hash 目录。
# 训练超参（lr / epochs 之类）不进 hash —— 它们不改变最终的码本语义契约。
_HASH_KEYS = (
    "method",
    "implementation",
    "levels",
    "codebook_size",
    "layer_prefixes",
    "collision_handling",
    "encoder.model",
    "encoder.max_length",
    "encoder.pooling",
    "rqvae.pca_dim",
    "rqvae.latent_dim",
    "rqvae.e_dim",
    "rqvae.layers",
    "rqvae.num_emb_list",
    "rqkmeans.pca_dim",
    "rqkmeans.enforce_unique",
)


def _dig(cfg: dict[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def items_fingerprint(item_ids: Sequence[str]) -> str:
    """物品集合的指纹。物品增删 → SID 必须重建，这里就能挡住。"""
    h = hashlib.sha256()
    h.update(str(len(item_ids)).encode())
    for item in sorted(map(str, item_ids)):
        h.update(item.encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def config_hash(
    sid_cfg: dict[str, Any],
    *,
    dataset: str,
    seed: int,
    items_fp: str,
) -> str:
    """算出这份 SID 配置的短 hash，用作产物目录名。"""
    payload = {key: _dig(sid_cfg, key) for key in _HASH_KEYS}
    payload["dataset"] = dataset
    payload["seed"] = int(seed)
    payload["items"] = items_fp
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def artifact_dir(
    sid_cfg: dict[str, Any],
    *,
    dataset: str,
    seed: int,
    items_fp: str,
    root: Path | None = None,
) -> Path:
    base = Path(root or sid_cfg.get("artifacts_root") or "artifacts/sid")
    return base / dataset / config_hash(sid_cfg, dataset=dataset, seed=seed, items_fp=items_fp)


@dataclass
class SidManifest:
    """描述一份已生成的 SID 产物。"""

    config_hash: str
    dataset: str
    seed: int
    items_fingerprint: str
    method: str
    levels: int
    codebook_size: int
    layer_prefixes: list[str]
    n_items: int
    collision_rate: float
    encoder: str
    created_at: str
    sid_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_hash": self.config_hash,
            "dataset": self.dataset,
            "seed": self.seed,
            "items_fingerprint": self.items_fingerprint,
            "method": self.method,
            "levels": self.levels,
            "codebook_size": self.codebook_size,
            "layer_prefixes": list(self.layer_prefixes),
            "n_items": self.n_items,
            "collision_rate": self.collision_rate,
            "encoder": self.encoder,
            "created_at": self.created_at,
            "sid_config": self.sid_config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SidManifest:
        return cls(
            config_hash=str(data["config_hash"]),
            dataset=str(data["dataset"]),
            seed=int(data["seed"]),
            items_fingerprint=str(data["items_fingerprint"]),
            method=str(data["method"]),
            levels=int(data["levels"]),
            codebook_size=int(data["codebook_size"]),
            layer_prefixes=list(data.get("layer_prefixes") or []),
            n_items=int(data["n_items"]),
            collision_rate=float(data.get("collision_rate", 0.0)),
            encoder=str(data.get("encoder") or ""),
            created_at=str(data.get("created_at") or ""),
            sid_config=dict(data.get("sid_config") or {}),
        )


def write_manifest(
    out_dir: Path,
    *,
    sid_cfg: dict[str, Any],
    dataset: str,
    seed: int,
    items_fp: str,
    n_items: int,
    collision_rate: float,
) -> SidManifest:
    manifest = SidManifest(
        config_hash=config_hash(sid_cfg, dataset=dataset, seed=seed, items_fp=items_fp),
        dataset=dataset,
        seed=int(seed),
        items_fingerprint=items_fp,
        method=str(sid_cfg.get("method") or "rqvae"),
        levels=int(sid_cfg.get("levels") or 3),
        codebook_size=int(sid_cfg.get("codebook_size") or 256),
        layer_prefixes=list(sid_cfg.get("layer_prefixes") or ["a", "b", "c"]),
        n_items=int(n_items),
        collision_rate=float(collision_rate),
        encoder=str(_dig(sid_cfg, "encoder.model") or ""),
        created_at=datetime.now(timezone.utc).isoformat(),
        sid_config=dict(sid_cfg),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST_NAME
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_manifest(sid_path: Path) -> SidManifest:
    path = Path(sid_path) / MANIFEST_NAME
    if not path.is_file():
        raise MissingArtifactError(
            f"{sid_path} 里没有 {MANIFEST_NAME}。SID 产物必须由 "
            f"`bash prepare.sh`（STEPS=sid）生成，不接受手工拼出来的目录。"
        )
    with path.open(encoding="utf-8") as fh:
        return SidManifest.from_dict(json.load(fh))


def resolve_sid_dir(
    sid_cfg: dict[str, Any],
    *,
    dataset: str,
    seed: int,
    item_ids: Sequence[str],
    root: Path | None = None,
    strict: bool = True,
) -> Path:
    """定位并校验 SID 产物目录。

    两种来源，优先级从高到低：

      1. ``sid.import_from: <目录>`` —— **显式导入一份现成的 SID**。
         典型场景：在本地/某台机器上建好 SID，打包拷到训练机直接用，
         不用在每台机器上重跑一遍 RQ-VAE（要 GPU、要几十分钟）。
         导入的产物**照样校验物品指纹**，对不上就报错。

      2. 按配置 hash 在 ``artifacts/sid/<dataset>/<hash>/`` 下找。

    ``strict=True``（训练/评测时的默认）下，产物不存在或指纹对不上就抛错，
    **绝不重建** —— 旧版"train 时发现缺了就补建"会让不同 run 悄悄用上不同的
    SID 表，而 SID 一变，物品语义前缀全变，bias 指标根本不可比。
    """
    items_fp = items_fingerprint(item_ids)

    imported = sid_cfg.get("import_from")
    if imported:
        path = Path(str(imported))
        if not path.is_dir():
            raise MissingArtifactError(
                f"sid.import_from 指向的目录不存在：{path}\n"
                f"  这个字段用于导入预先建好的 SID（省掉在训练机上重跑 RQ-VAE）。\n"
                f"  留空则按配置 hash 在 {root or 'artifacts/sid'} 下查找。"
            )
        _validate_sid_dir(path, items_fp=items_fp, item_ids=item_ids, imported=True)
        return path

    path = artifact_dir(sid_cfg, dataset=dataset, seed=seed, items_fp=items_fp, root=root)
    if not path.is_dir():
        if not strict:
            return path
        available = _list_available(path.parent)
        raise MissingArtifactError(
            f"找不到 SID 产物：{path}\n"
            f"  当前配置 hash = {config_hash(sid_cfg, dataset=dataset, seed=seed, items_fp=items_fp)}\n"
            f"  物品集合指纹  = {items_fp}（{len(item_ids)} 个物品）\n"
            f"{available}"
            f"\n两个办法：\n"
            f"  a) 本机构建： STEPS=sid bash prepare.sh\n"
            f"  b) 导入现成的：在配置里写 sid.import_from: /path/to/sid_dir\n"
            f"     （或命令行 bash run.sh sid.import_from=/path/to/sid_dir）"
        )

    _validate_sid_dir(path, items_fp=items_fp, item_ids=item_ids, imported=False)
    return path


def _validate_sid_dir(
    path: Path, *, items_fp: str, item_ids: Sequence[str], imported: bool
) -> None:
    manifest = load_manifest(path)
    if manifest.items_fingerprint != items_fp:
        hint = (
            "导入的 SID 是在别的数据集/别的过滤参数上建的。"
            if imported
            else "物品集合变了，SID 必须重建：STEPS=sid FORCE=1 bash prepare.sh"
        )
        raise ConfigurationError(
            f"SID 产物 {path} 与当前数据集对不上：\n"
            f"  产物指纹 = {manifest.items_fingerprint}（{manifest.n_items} 个物品）\n"
            f"  当前指纹 = {items_fp}（{len(item_ids)} 个物品）\n"
            f"{hint}"
        )
    if not (path / SID_MAP_NAME).is_file():
        raise MissingArtifactError(f"SID 产物 {path} 缺少 {SID_MAP_NAME}")


def _list_available(parent: Path) -> str:
    """报错时把这个数据集下已有的 SID 产物列出来，方便直接 import_from。"""
    if not parent.is_dir():
        return ""
    entries = [d for d in sorted(parent.iterdir()) if (d / MANIFEST_NAME).is_file()]
    if not entries:
        return ""
    lines = [f"\n  该数据集下已有 {len(entries)} 份 SID 产物："]
    for d in entries:
        try:
            m = load_manifest(d)
            lines.append(
                f"    {d}  (method={m.method} levels={m.levels} K={m.codebook_size} "
                f"items={m.n_items} 指纹={m.items_fingerprint})"
            )
        except Exception:  # noqa: BLE001
            lines.append(f"    {d}  (manifest 读取失败)")
    return "\n".join(lines) + "\n"
