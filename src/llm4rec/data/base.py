"""数据集适配器抽象 + 注册表 —— 换数据集只改 config，不碰代码。

任何数据集只要实现这个接口，就能被三条路线直接用：

    class MyDataset(DatasetAdapter):
        name = "mydata"
        def preprocess(self, cfg, force): ...

产物契约（所有适配器都必须产出这四个文件，位置由 ``processed_dir`` 决定）：

    interactions.jsonl   user_id / item_id / timestamp / rating / split
    item_meta.json       item_id → {title, description, categories, brand, text}
    popularity.json      counts；★ 只统计 train split
    stats.json           规模统计

只要这个契约满足，SID 构建、BM25 索引、样本构建、bias 评测全部不用改 ——
它们只认契约，不认具体数据集。

★ ``item_meta`` 里的 ``text`` 字段是三处共用的物品表示：RQ-VAE 的编码输入、
  BM25 的索引正文、reranker 的知识来源。必须由适配器统一产出，
  不能各自再拼一遍 —— 否则三条路线"看到"的物品其实不是同一个。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from llm4rec.core.exceptions import ConfigurationError, MissingArtifactError


@dataclass
class ProcessedPaths:
    root: Path
    interactions: Path
    item_meta: Path
    popularity: Path
    stats: Path

    @classmethod
    def of(cls, processed_dir: Path) -> ProcessedPaths:
        root = Path(processed_dir)
        return cls(
            root=root,
            interactions=root / "interactions.jsonl",
            item_meta=root / "item_meta.json",
            popularity=root / "popularity.json",
            stats=root / "stats.json",
        )

    def exists(self) -> bool:
        return all(
            p.is_file()
            for p in (self.interactions, self.item_meta, self.popularity, self.stats)
        )


class DatasetAdapter(ABC):
    """把某个原始语料转成统一契约。"""

    name: str = ""

    # ---------------------------------------------------------------- 必须实现
    @abstractmethod
    def preprocess(self, cfg: dict[str, Any], *, force: bool = False) -> ProcessedPaths:
        """读原始文件 → 写出四件套产物。"""

    @abstractmethod
    def raw_files(self, cfg: dict[str, Any]) -> dict[str, tuple[str, Path]]:
        """需要哪些原始文件：``{说明: (下载 URL, 本地落点)}``。

        ``download()`` 和"缺文件时的报错信息"都从这里来 —— 只写一遍，
        不会出现"报错说要 A 文件，下载脚本却去下 B"的不一致。
        """

    # ---------------------------------------------------------------- 下载
    def download(self, cfg: dict[str, Any], *, force: bool = False) -> None:
        """按 ``raw_files()`` 下载原始数据。带断点续传和大小校验。"""
        for label, (url, dest) in self.raw_files(cfg).items():
            _download_one(label, url, Path(dest), force=force)

    def check_raw(self, cfg: dict[str, Any]) -> list[str]:
        """返回缺失的原始文件说明列表（空 = 齐全）。"""
        missing = []
        for label, (_, dest) in self.raw_files(cfg).items():
            path = Path(dest)
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(f"{label} → {path}")
        return missing

    # ---------------------------------------------------------------- 通用实现
    def processed_dir(self, cfg: dict[str, Any]) -> Path:
        data = cfg["data"]
        base = Path(data["processed_dir"])
        variant = data.get("category") or data.get("variant")
        return base / str(variant) if variant else base

    def dataset_key(self, cfg: dict[str, Any]) -> str:
        """用于产物目录命名（SID / BM25 / embedding 都按它分目录）。"""
        data = cfg["data"]
        variant = data.get("category") or data.get("variant")
        return f"{data['name']}_{variant}" if variant else str(data["name"])

    def load_item_meta(self, cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
        path = ProcessedPaths.of(self.processed_dir(cfg)).item_meta
        if not path.is_file():
            raise MissingArtifactError(f"缺少 {path}，先跑 `STEPS=data bash prepare.sh`")
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)

    def load_interactions(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        path = ProcessedPaths.of(self.processed_dir(cfg)).interactions
        if not path.is_file():
            raise MissingArtifactError(f"缺少 {path}，先跑 `STEPS=data bash prepare.sh`")
        with path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    @staticmethod
    def user_sequences(
        interactions: Iterable[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in interactions:
            by_user[str(row["user_id"])].append(row)
        for events in by_user.values():
            events.sort(key=lambda r: (r["timestamp"], r["item_id"]))
        return dict(by_user)


# ------------------------------------------------------------------ 注册表

_REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register_dataset(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
    if not cls.name:
        raise ConfigurationError(f"{cls.__name__} 必须设置 name")
    _REGISTRY[cls.name] = cls
    return cls


def get_adapter(cfg: dict[str, Any]) -> DatasetAdapter:
    """按 ``data.name`` 取适配器 —— 这就是数据集热插拔的入口。"""
    _ensure_loaded()
    name = str(cfg["data"]["name"])
    if name not in _REGISTRY:
        raise ConfigurationError(
            f"未知数据集 '{name}'。已注册：{sorted(_REGISTRY)}\n"
            f"新增数据集：在 llm4rec/data/ 下实现 DatasetAdapter 并加 @register_dataset，"
            f"再写一个 configs/data/<name>.yaml。"
        )
    return _REGISTRY[name]()


def available_datasets() -> list[str]:
    _ensure_loaded()
    return sorted(_REGISTRY)


def _ensure_loaded() -> None:
    """惰性导入内置适配器，避免循环 import。"""
    if _REGISTRY:
        return
    from llm4rec.data import amazon23, movielens  # noqa: F401


# ------------------------------------------------------------------ 共享工具


def build_item_text(meta: dict[str, Any], fields: list[str], max_chars: int) -> str:
    """拼出物品的自然语言表示（RQ-VAE / BM25 / reranker 共用）。"""
    parts: list[str] = []
    for field in fields:
        value = meta.get(field)
        if not value:
            continue
        if isinstance(value, list):
            value = " ".join(str(v) for v in value if v)
        text = str(value).strip()
        if text:
            parts.append(text)
    return " ".join(parts)[:max_chars]


def iterative_kcore(
    interactions: list[dict[str, Any]], min_uc: int, min_sc: int
) -> list[dict[str, Any]]:
    """反复过滤直到同时满足 user/item 的最小交互数。"""
    from collections import Counter

    rows = interactions
    while True:
        user_counts = Counter(r["user_id"] for r in rows)
        item_counts = Counter(r["item_id"] for r in rows)
        kept = [
            r
            for r in rows
            if user_counts[r["user_id"]] >= min_uc and item_counts[r["item_id"]] >= min_sc
        ]
        if len(kept) == len(rows):
            return kept
        if not kept:
            raise ConfigurationError(
                f"k-core 过滤后没有数据了（min_uc={min_uc}, min_sc={min_sc}），把阈值调小"
            )
        rows = kept


def leave_one_out_split(
    interactions: list[dict[str, Any]], *, min_history: int = 2
) -> list[dict[str, Any]]:
    """按时间序：倒数第二条 → val，最后一条 → test，其余 train。"""
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        by_user[row["user_id"]].append(row)

    out: list[dict[str, Any]] = []
    for events in by_user.values():
        events.sort(key=lambda r: (r["timestamp"], r["item_id"]))
        if len(events) < min_history + 2:
            continue
        for row in events[:-2]:
            row["split"] = "train"
        events[-2]["split"] = "val"
        events[-1]["split"] = "test"
        out.extend(events)
    return out


def write_artifacts(
    paths: ProcessedPaths,
    *,
    interactions: list[dict[str, Any]],
    item_meta: dict[str, dict[str, Any]],
    stats: dict[str, Any],
) -> None:
    """写四件套。流行度 ★ 只统计 train split。"""
    from collections import Counter

    all_items = sorted({r["item_id"] for r in interactions})
    train_counts = Counter(r["item_id"] for r in interactions if r["split"] == "train")

    paths.root.mkdir(parents=True, exist_ok=True)
    with paths.interactions.open("w", encoding="utf-8") as fh:
        for row in interactions:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    _dump(paths.item_meta, {i: item_meta[i] for i in all_items if i in item_meta})
    _dump(
        paths.popularity,
        {
            "counts": {i: int(train_counts.get(i, 0)) for i in all_items},
            # 混进 test 的话，"模型偏向热门"和"热门本来就更容易命中"就分不开了
            "source": "train_only",
        },
    )
    stats = dict(stats)
    stats.update(
        {
            "n_users": len({r["user_id"] for r in interactions}),
            "n_items": len(all_items),
            "n_interactions": len(interactions),
            "n_train": sum(1 for r in interactions if r["split"] == "train"),
            "n_val": sum(1 for r in interactions if r["split"] == "val"),
            "n_test": sum(1 for r in interactions if r["split"] == "test"),
            "popularity_source": "train_only",
        }
    )
    _dump(paths.stats, stats)


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# ------------------------------------------------------------------ 下载工具


def _download_one(label: str, url: str, dest: Path, *, force: bool = False) -> None:
    """下载单个文件，带断点续传 + 完整性校验。

    Amazon23 单个类目就有几百 MB，网络断一次重头下很浪费，所以走 ``.part``
    临时文件 + Range 续传，下完比对 Content-Length 再改名 —— 半截文件永远
    不会以正式名字存在，下次跑不会把残缺文件当成功。
    """
    import shutil
    import urllib.request

    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        print(f"[download] 已存在，跳过：{dest.name}（{dest.stat().st_size / 1048576:.1f} MB）")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    resume_from = part.stat().st_size if part.is_file() and not force else 0
    if force and part.is_file():
        part.unlink()
        resume_from = 0

    request = urllib.request.Request(url, headers={"User-Agent": "llm4rec-bias/1.0"})
    if resume_from:
        request.add_header("Range", f"bytes={resume_from}-")
        print(f"[download] {label}：从 {resume_from / 1048576:.1f} MB 处续传")

    print(f"[download] {label}\n           {url}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            declared = response.headers.get("Content-Length")
            total = (int(declared) + resume_from) if declared else None
            if total:
                print(f"           大小 {total / 1048576:.1f} MB → {dest}")
            mode = "ab" if resume_from else "wb"
            downloaded = resume_from
            last_report = 0
            with part.open(mode) as fh:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total and downloaded - last_report > (1 << 24):  # 每 16MB 报一次
                        last_report = downloaded
                        print(
                            f"           {downloaded / 1048576:7.1f} / "
                            f"{total / 1048576:.1f} MB  ({100 * downloaded / total:.0f}%)",
                            flush=True,
                        )
    except Exception as exc:  # noqa: BLE001
        raise MissingArtifactError(
            f"下载 {label} 失败：{exc}\n"
            f"  URL: {url}\n"
            f"  已下载的部分保留在 {part}，重跑会续传。\n"
            f"  也可以手动下载后放到 {dest}"
        ) from exc

    if total and part.stat().st_size != total:
        raise MissingArtifactError(
            f"{label} 下载不完整：{part.stat().st_size} / {total} 字节。"
            f"重跑 `STEPS=download bash prepare.sh` 会续传。"
        )
    shutil.move(str(part), str(dest))
    print(f"[download] 完成 {dest.name}（{dest.stat().st_size / 1048576:.1f} MB）")


def extract_archive(archive: Path, dest_dir: Path, *, strip_top_level: bool = True) -> None:
    """解压 zip/tar 到 ``dest_dir``，可选剥掉顶层目录。"""
    import shutil
    import tarfile
    import tempfile
    import zipfile

    archive, dest_dir = Path(archive), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(tmp_path)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tf:
                tf.extractall(tmp_path)
        else:
            raise ConfigurationError(f"不认识的压缩格式：{archive}")

        entries = list(tmp_path.iterdir())
        source = entries[0] if (strip_top_level and len(entries) == 1 and entries[0].is_dir()) else tmp_path
        for item in source.iterdir():
            target = dest_dir / item.name
            if target.exists():
                continue
            shutil.move(str(item), str(target))
    print(f"[download] 解压完成 → {dest_dir}")
