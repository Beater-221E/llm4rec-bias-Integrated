"""MovieLens-100K adapter."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

from llm4rec_bias_Integrated.core.exceptions import DatasetValidationError, MissingArtifactError
from llm4rec_bias_Integrated.core.schemas import Interaction
from llm4rec_bias_Integrated.datasets.movielens.common import ItemMetadata
from llm4rec_bias_Integrated.datasets.movielens.metadata import (
    genres_from_ml100k_flags,
    parse_title_year,
)
from llm4rec_bias_Integrated.datasets.movielens.preprocess import MovieLensAdapterBase
from llm4rec_bias_Integrated.datasets.registry import register_dataset

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


@register_dataset("movielens_100k")
class MovieLens100KAdapter(MovieLensAdapterBase):
    """GroupLens MovieLens 100K (u.data / u.item)."""

    name = "movielens_100k"
    dataset_slug = "movielens_100k"

    def _ensure_raw(self) -> Path:
        raw = self.raw_dir / "ml-100k"
        if (raw / "u.data").exists() and not self.force_download:
            return raw
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        zpath = self.raw_dir / "ml-100k.zip"
        if not zpath.exists() or self.force_download:
            urllib.request.urlretrieve(ML100K_URL, zpath)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(self.raw_dir)
        if not (raw / "u.data").exists():
            raise MissingArtifactError(f"Expected u.data under {raw}")
        return raw

    def _parse_raw(self, raw_path: Path) -> tuple[list[Interaction], dict[str, ItemMetadata]]:
        item_path = raw_path / "u.item"
        data_path = raw_path / "u.data"
        if not item_path.exists() or not data_path.exists():
            raise MissingArtifactError(f"Missing u.item/u.data in {raw_path}")

        meta: dict[str, ItemMetadata] = {}
        with item_path.open(encoding="latin-1") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 5:
                    continue
                item_id = str(int(parts[0]))
                title_raw = parts[1]
                title, year = parse_title_year(title_raw)
                genres = genres_from_ml100k_flags(parts[5:])
                meta[item_id] = ItemMetadata(
                    item_id=item_id,
                    title=title_raw.strip(),
                    genres=genres,
                    release_year=year,
                    raw={"title_clean": title},
                )

        interactions: list[Interaction] = []
        with data_path.open(encoding="utf-8") as fh:
            for line in fh:
                user_s, item_s, rating_s, ts_s = line.strip().split("\t")
                rating = float(rating_s)
                if rating < self.rating_threshold:
                    continue
                item_id = str(int(item_s))
                if item_id not in meta:
                    continue
                interactions.append(
                    Interaction(
                        user_id=str(int(user_s)),
                        item_id=item_id,
                        rating=rating,
                        timestamp=int(ts_s),
                    )
                )
        if not interactions:
            raise DatasetValidationError("MovieLens-100K produced zero interactions")
        return interactions, meta
