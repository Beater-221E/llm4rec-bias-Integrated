"""Amazon Reviews 2023 适配器。

原始文件（https://amazon-reviews-2023.github.io/）：
    {Category}.jsonl.gz        评论；user_id / parent_asin / timestamp(毫秒) / rating
    meta_{Category}.jsonl.gz   物品元数据；keyed by parent_asin，description 是 list
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator

from llm4rec.core.exceptions import ConfigurationError, MissingArtifactError
from llm4rec.data.base import (
    DatasetAdapter,
    ProcessedPaths,
    build_item_text,
    iterative_kcore,
    leave_one_out_split,
    register_dataset,
    write_artifacts,
)


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


@register_dataset
class Amazon23Adapter(DatasetAdapter):
    name = "amazon23"

    # 官方托管（https://amazon-reviews-2023.github.io/）
    BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"

    def raw_files(self, cfg: dict[str, Any]) -> dict[str, tuple[str, Path]]:
        data = cfg["data"]
        category = str(data["category"])
        raw_dir = Path(data["raw_dir"])
        base = str(data.get("download_base_url") or self.BASE_URL).rstrip("/")
        return {
            f"{category} 评论": (
                f"{base}/review_categories/{category}.jsonl.gz",
                raw_dir / f"{category}.jsonl.gz",
            ),
            f"{category} 元数据": (
                f"{base}/meta_categories/meta_{category}.jsonl.gz",
                raw_dir / f"meta_{category}.jsonl.gz",
            ),
        }

    def resolve_raw_files(self, category: str, raw_dir: Path) -> tuple[Path, Path]:
        reviews = [
            raw_dir / f"{category}.jsonl.gz",
            raw_dir / f"{category}.jsonl",
            raw_dir / f"{category}_5.json.gz",
        ]
        metas = [
            raw_dir / f"meta_{category}.jsonl.gz",
            raw_dir / f"meta_{category}.jsonl",
            raw_dir / f"meta_{category}.json.gz",
        ]
        review = next((p for p in reviews if p.is_file() and p.stat().st_size), None)
        meta = next((p for p in metas if p.is_file() and p.stat().st_size), None)
        if review is None or meta is None:
            raise MissingArtifactError(
                f"在 {raw_dir} 下找不到 {category} 的原始文件。\n"
                f"  评论候选  ：{[p.name for p in reviews]}\n"
                f"  元数据候选：{[p.name for p in metas]}\n"
                f"跑 `STEPS=download bash prepare.sh` 自动下载。"
            )
        return review, meta

    def preprocess(self, cfg: dict[str, Any], *, force: bool = False) -> ProcessedPaths:
        data = cfg["data"]
        category = str(data["category"])
        paths = ProcessedPaths.of(self.processed_dir(cfg))

        if paths.exists() and not force:
            print(f"[data] 已存在，跳过：{paths.root}（FORCE=1 可强制重建）")
            return paths

        review_path, meta_path = self.resolve_raw_files(category, Path(data["raw_dir"]))
        fields = list(data.get("item_text_fields") or ["title", "description"])
        max_chars = int(data.get("item_text_max_chars") or 512)
        threshold = float(data.get("rating_threshold") or 4.0)

        print(f"[data] 读评论 {review_path.name} …")
        interactions: list[dict[str, Any]] = []
        for row in iter_jsonl(review_path):
            rating = row.get("rating")
            if rating is None or float(rating) < threshold:
                continue
            user = row.get("user_id")
            item = row.get("parent_asin") or row.get("asin")
            ts = row.get("timestamp")
            if not user or not item or ts is None:
                continue
            interactions.append(
                {
                    "user_id": str(user),
                    "item_id": str(item),
                    "timestamp": int(int(ts) // 1000),  # 毫秒 → 秒
                    "rating": float(rating),
                }
            )
        print(f"[data]   正样本交互 {len(interactions)}")

        if bool(data.get("iterative_kcore", True)):
            interactions = iterative_kcore(
                interactions, int(data.get("min_uc") or 5), int(data.get("min_sc") or 5)
            )
            print(f"[data]   k-core 后 {len(interactions)}")

        keep = {r["item_id"] for r in interactions}
        print(f"[data] 读元数据 {meta_path.name} …")
        item_meta: dict[str, dict[str, Any]] = {}
        for row in iter_jsonl(meta_path):
            item = row.get("parent_asin") or row.get("asin")
            if item and str(item) in keep:
                item_meta[str(item)] = self._normalize(row, fields, max_chars)

        missing = keep - set(item_meta)
        if missing:
            print(f"[data]   丢弃 {len(missing)} 个缺元数据的物品")
            interactions = [r for r in interactions if r["item_id"] in item_meta]

        final = leave_one_out_split(
            interactions, min_history=int(data.get("min_history") or 2)
        )
        if not final:
            raise ConfigurationError("划分后没有数据 —— 检查 min_uc / min_history")

        write_artifacts(
            paths,
            interactions=final,
            item_meta=item_meta,
            stats={
                "dataset": self.name,
                "category": category,
                "rating_threshold": threshold,
                "min_uc": int(data.get("min_uc") or 5),
                "min_sc": int(data.get("min_sc") or 5),
            },
        )
        print(f"[data] 完成 → {paths.root}")
        with paths.stats.open(encoding="utf-8") as fh:
            for key, value in json.load(fh).items():
                print(f"[data]   {key:18s} = {value}")
        return paths

    @staticmethod
    def _normalize(row: dict[str, Any], fields: list[str], max_chars: int) -> dict[str, Any]:
        desc = row.get("description")
        if isinstance(desc, list):
            desc = " ".join(str(d) for d in desc if d)
        cats = row.get("categories") or row.get("category")
        if isinstance(cats, list):
            cats = " > ".join(str(c) for c in cats if c)
        entry = {
            "title": str(row.get("title") or "").strip(),
            "description": str(desc or "").strip(),
            "categories": str(cats or "").strip(),
            "brand": str(row.get("store") or row.get("brand") or "").strip(),
        }
        entry["text"] = build_item_text(entry, fields, max_chars)
        return entry
