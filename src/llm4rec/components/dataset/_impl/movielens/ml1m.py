"""MovieLens-1M adapter."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

from llm4rec.core.exceptions import DatasetValidationError, MissingArtifactError
from llm4rec.core.schemas import Interaction
from llm4rec.components.dataset._impl.movielens.common import ItemMetadata
from llm4rec.components.dataset._impl.movielens.metadata import (
    genres_from_pipe_string,
    parse_title_year,
)
from llm4rec.components.dataset._impl.movielens.preprocess import MovieLensAdapterBase
from llm4rec.components.dataset._impl.registry import register_dataset

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


@register_dataset("movielens_1m")
class MovieLens1MAdapter(MovieLensAdapterBase):
    """GroupLens MovieLens 1M (ratings.dat / movies.dat)."""

    name = "movielens_1m"
    dataset_slug = "movielens_1m"

    def _ensure_raw(self) -> Path:
        raw = self.raw_dir / "ml-1m"
        if (raw / "ratings.dat").exists() and not self.force_download:
            return raw
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        zpath = self.raw_dir / "ml-1m.zip"
        if not zpath.exists() or self.force_download:
            urllib.request.urlretrieve(ML1M_URL, zpath)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(self.raw_dir)
        if not (raw / "ratings.dat").exists():
            raise MissingArtifactError(f"Expected ratings.dat under {raw}")
        return raw

    def _parse_raw(self, raw_path: Path) -> tuple[list[Interaction], dict[str, ItemMetadata]]:
        movies_path = raw_path / "movies.dat"
        ratings_path = raw_path / "ratings.dat"
        if not movies_path.exists() or not ratings_path.exists():
            raise MissingArtifactError(f"Missing movies.dat/ratings.dat in {raw_path}")

        meta: dict[str, ItemMetadata] = {}
        with movies_path.open(encoding="latin-1") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("::")
                if len(parts) < 3:
                    continue
                item_id = str(int(parts[0]))
                title_raw = parts[1]
                title, year = parse_title_year(title_raw)
                genres = genres_from_pipe_string(parts[2])
                meta[item_id] = ItemMetadata(
                    item_id=item_id,
                    title=title_raw.strip(),
                    genres=genres,
                    release_year=year,
                    raw={"title_clean": title},
                )

        interactions: list[Interaction] = []
        with ratings_path.open(encoding="latin-1") as fh:
            for line in fh:
                user_s, item_s, rating_s, ts_s = line.strip().split("::")
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
            raise DatasetValidationError("MovieLens-1M produced zero interactions")
        return interactions, meta
