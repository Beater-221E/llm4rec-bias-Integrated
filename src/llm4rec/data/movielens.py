"""MovieLens 适配器（ML-1M / ML-100K）。

存在的意义有两个：
  1. DPO4Rec 原文用的就是 ML-1M，需要它才能和论文数字对齐；
  2. **证明数据集是真的热插拔** —— 加这个文件 + 一个 configs/data/movielens.yaml，
     三条路线、SID 构建、BM25、bias 评测全部零改动就能用。

原始文件（https://grouplens.org/datasets/movielens/）：
    ML-1M   : ratings.dat / movies.dat  （``::`` 分隔，latin-1）
    ML-100K : u.data / u.item           （制表符 / ``|`` 分隔）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm4rec.core.exceptions import ConfigurationError, MissingArtifactError
from llm4rec.data.base import (
    DatasetAdapter,
    ProcessedPaths,
    build_item_text,
    extract_archive,
    iterative_kcore,
    leave_one_out_split,
    register_dataset,
    write_artifacts,
)


@register_dataset
class MovieLensAdapter(DatasetAdapter):
    name = "movielens"

    BASE_URL = "https://files.grouplens.org/datasets/movielens"
    # variant → (压缩包名, 解压后需要的两个文件)
    VARIANTS = {
        "ml-1m": ("ml-1m.zip", ("ratings.dat", "movies.dat")),
        "ml-100k": ("ml-100k.zip", ("u.data", "u.item")),
    }

    def _variant(self, cfg: dict[str, Any]) -> str:
        data = cfg["data"]
        name = str(data.get("variant") or data.get("category") or "ml-1m").lower()
        name = {"ml1m": "ml-1m", "ml100k": "ml-100k"}.get(name, name)
        if name not in self.VARIANTS:
            raise ConfigurationError(
                f"未知 MovieLens 变体 '{name}'（支持 {sorted(self.VARIANTS)}）"
            )
        return name

    def raw_files(self, cfg: dict[str, Any]) -> dict[str, tuple[str, Path]]:
        variant = self._variant(cfg)
        archive, _ = self.VARIANTS[variant]
        raw_dir = Path(cfg["data"]["raw_dir"])
        base = str(cfg["data"].get("download_base_url") or self.BASE_URL).rstrip("/")
        return {f"MovieLens {variant}": (f"{base}/{archive}", raw_dir / archive)}

    def download(self, cfg: dict[str, Any], *, force: bool = False) -> None:
        """MovieLens 是 zip，下完还要解压。"""
        super().download(cfg, force=force)
        variant = self._variant(cfg)
        archive_name, needed = self.VARIANTS[variant]
        raw_dir = Path(cfg["data"]["raw_dir"])
        if all((raw_dir / f).is_file() for f in needed) and not force:
            print(f"[download] {variant} 已解压，跳过")
            return
        extract_archive(raw_dir / archive_name, raw_dir, strip_top_level=True)

    def check_raw(self, cfg: dict[str, Any]) -> list[str]:
        """MovieLens 检查的是**解压后**的文件，不是 zip。"""
        variant = self._variant(cfg)
        _, needed = self.VARIANTS[variant]
        raw_dir = Path(cfg["data"]["raw_dir"])
        return [
            f"MovieLens {variant} 的 {f} → {raw_dir / f}"
            for f in needed
            if not (raw_dir / f).is_file()
        ]

    def preprocess(self, cfg: dict[str, Any], *, force: bool = False) -> ProcessedPaths:
        data = cfg["data"]
        variant = self._variant(cfg)
        paths = ProcessedPaths.of(self.processed_dir(cfg))

        if paths.exists() and not force:
            print(f"[data] 已存在，跳过：{paths.root}（FORCE=1 可强制重建）")
            return paths

        raw_dir = Path(data["raw_dir"])
        fields = list(data.get("item_text_fields") or ["title", "categories"])
        max_chars = int(data.get("item_text_max_chars") or 512)
        threshold = float(data.get("rating_threshold") or 4.0)

        _, needed = self.VARIANTS[variant]
        ratings, movies = raw_dir / needed[0], raw_dir / needed[1]
        reader = self._read_ml1m if variant == "ml-1m" else self._read_ml100k
        for path in (ratings, movies):
            if not path.is_file():
                raise MissingArtifactError(
                    f"缺少 {path}。跑 `STEPS=download bash prepare.sh` 自动下载解压。"
                )

        interactions, item_meta = reader(ratings, movies, threshold, fields, max_chars)
        print(f"[data]   正样本交互 {len(interactions)}")

        if bool(data.get("iterative_kcore", True)):
            interactions = iterative_kcore(
                interactions, int(data.get("min_uc") or 5), int(data.get("min_sc") or 5)
            )
            print(f"[data]   k-core 后 {len(interactions)}")

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
                "variant": variant,
                "rating_threshold": threshold,
            },
        )
        print(f"[data] 完成 → {paths.root}")
        with paths.stats.open(encoding="utf-8") as fh:
            for key, value in json.load(fh).items():
                print(f"[data]   {key:18s} = {value}")
        return paths

    # ------------------------------------------------------------- 具体读取
    @staticmethod
    def _read_ml1m(
        ratings: Path, movies: Path, threshold: float, fields: list[str], max_chars: int
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        interactions = []
        with ratings.open(encoding="latin-1") as fh:
            for line in fh:
                parts = line.strip().split("::")
                if len(parts) != 4 or float(parts[2]) < threshold:
                    continue
                interactions.append(
                    {
                        "user_id": parts[0],
                        "item_id": parts[1],
                        "timestamp": int(parts[3]),
                        "rating": float(parts[2]),
                    }
                )
        meta: dict[str, dict[str, Any]] = {}
        with movies.open(encoding="latin-1") as fh:
            for line in fh:
                parts = line.strip().split("::")
                if len(parts) != 3:
                    continue
                entry = {
                    "title": parts[1],
                    "categories": parts[2].replace("|", ", "),
                    "description": "",
                    "brand": "",
                }
                entry["text"] = build_item_text(entry, fields, max_chars)
                meta[parts[0]] = entry
        return interactions, meta

    @staticmethod
    def _read_ml100k(
        ratings: Path, movies: Path, threshold: float, fields: list[str], max_chars: int
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        GENRES = [
            "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
        ]
        interactions = []
        with ratings.open(encoding="latin-1") as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) != 4 or float(parts[2]) < threshold:
                    continue
                interactions.append(
                    {
                        "user_id": parts[0],
                        "item_id": parts[1],
                        "timestamp": int(parts[3]),
                        "rating": float(parts[2]),
                    }
                )
        meta: dict[str, dict[str, Any]] = {}
        with movies.open(encoding="latin-1") as fh:
            for line in fh:
                parts = line.strip().split("|")
                if len(parts) < 24:
                    continue
                flags = parts[5:24]
                genres = [g for g, f in zip(GENRES, flags) if f == "1"]
                entry = {
                    "title": parts[1],
                    "categories": ", ".join(genres) or "unknown",
                    "description": "",
                    "brand": "",
                }
                entry["text"] = build_item_text(entry, fields, max_chars)
                meta[parts[0]] = entry
        return interactions, meta
