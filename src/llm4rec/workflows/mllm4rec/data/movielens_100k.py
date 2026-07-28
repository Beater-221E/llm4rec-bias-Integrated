# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
#
# Original behavior is preserved unless explicitly documented.

"""MovieLens datasets for MLLM4Rec-compatible preprocessing.

- ``ml-100k``: official-compatible — downloads **ml-latest-small** (ratings.csv / movies.csv).
- ``ml-100k-classic``: extension — classic GroupLens ml-100k (u.data / u.item).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from llm4rec.workflows.mllm4rec.data._download import download_and_extract_zip_folder
from llm4rec.workflows.mllm4rec.data.base_dataset import BaseMovieLensDataset
from llm4rec.workflows.mllm4rec.data.constants import (
    CLASSIC_ML100K_CODE,
    CLASSIC_ML100K_URL,
    ML_LATEST_SMALL_URL,
    OFFICIAL_ML100K_CODE,
)

logger = logging.getLogger("llm4rec.workflows.mllm4rec._stack")


def clean_movielens_title_official(raw_title: str) -> str:
    """Official ``load_meta_dict`` title cleaning (ml-latest-small / movies.csv).

    Expects titles like ``Toy Story (1995)`` with a 7-char year suffix `` (YYYY)``.
    """
    title = raw_title[:-7]
    year = raw_title[-7:]
    title = re.sub(r"\(.*?\)", "", title).strip()
    if any(", " + x in title.lower()[-5:] for x in ["a", "an", "the"]):
        title_pre = title.split(", ")[:-1]
        title_post = title.split(", ")[-1]
        title_pre = ", ".join(title_pre)
        title = title_post + " " + title_pre
    return title + year


class ML100KDataset(BaseMovieLensDataset):
    """Official MLLM4Rec ``ml-100k`` (= GroupLens ml-latest-small)."""

    @classmethod
    def code(cls) -> str:
        return OFFICIAL_ML100K_CODE

    @classmethod
    def url(cls) -> str:
        return ML_LATEST_SMALL_URL

    @classmethod
    def all_raw_file_names(cls) -> list[str]:
        # Official list includes users.csv; current ml-latest-small ships links.csv
        # instead. We accept either layout if ratings.csv + movies.csv exist.
        return ["README", "movies.csv", "ratings.csv"]

    def maybe_download_raw_dataset(self) -> None:
        folder_path = self.raw_folder_path()
        required = ["movies.csv", "ratings.csv"]
        if folder_path.is_dir() and all(
            (folder_path / name).is_file() for name in required
        ):
            logger.info("Raw data already exists. Skip downloading (%s)", folder_path)
            return
        logger.info("Raw file doesn't exist. Downloading ml-latest-small...")
        download_and_extract_zip_folder(
            self.url(),
            folder_path,
            zip_content_is_folder=self.zip_file_content_is_folder(),
        )

    def load_ratings_df(self) -> pd.DataFrame:
        file_path = self.raw_folder_path() / "ratings.csv"
        df = pd.read_csv(file_path)
        df.columns = ["uid", "sid", "rating", "timestamp"]
        return df

    def load_meta_dict(self) -> dict[Any, str]:
        file_path = self.raw_folder_path() / "movies.csv"
        df = pd.read_csv(file_path, encoding="ISO-8859-1")
        meta_dict: dict[Any, str] = {}
        for row in df.itertuples():
            # row[1]=movieId, row[2]=title (itertuples: Index, movieId, title, genres)
            meta_dict[row[1]] = clean_movielens_title_official(str(row[2]))
        return meta_dict


class ML100KClassicDataset(BaseMovieLensDataset):
    """Extension: classic GroupLens MovieLens-100K (u.data / u.item).

    Not official-compatible with MLLM4Rec ``ml-100k`` runs.
    """

    @classmethod
    def code(cls) -> str:
        return CLASSIC_ML100K_CODE

    @classmethod
    def url(cls) -> str:
        return CLASSIC_ML100K_URL

    @classmethod
    def all_raw_file_names(cls) -> list[str]:
        return ["u.data", "u.item", "u.user", "u.genre"]

    def maybe_download_raw_dataset(self) -> None:
        folder_path = self.raw_folder_path()
        # Zip extracts to ml-100k/ subfolder; allow raw_dir to point at that folder.
        if folder_path.is_dir() and (folder_path / "u.data").is_file():
            logger.info("Raw classic ml-100k already exists. Skip downloading")
            return
        parent = folder_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        download_and_extract_zip_folder(
            self.url(),
            parent / "ml-100k",
            zip_content_is_folder=True,
        )
        # If cfg.raw_dir is not the extracted folder, still OK if user pointed correctly.
        if not (folder_path / "u.data").is_file():
            extracted = parent / "ml-100k"
            if extracted.is_dir() and folder_path.resolve() != extracted.resolve():
                logger.warning(
                    "Classic ml-100k extracted to %s; config raw_dir=%s",
                    extracted,
                    folder_path,
                )

    def load_ratings_df(self) -> pd.DataFrame:
        file_path = self.raw_folder_path() / "u.data"
        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            names=["uid", "sid", "rating", "timestamp"],
        )
        return df

    def load_meta_dict(self) -> dict[Any, str]:
        file_path = self.raw_folder_path() / "u.item"
        meta_dict: dict[Any, str] = {}
        with file_path.open(encoding="ISO-8859-1") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 2:
                    continue
                raw_id = int(parts[0])
                raw_title = parts[1]
                # Reuse official year-suffix cleaning when title ends with `` (YYYY)``.
                if len(raw_title) >= 7 and raw_title[-6:-1].isdigit() and raw_title.endswith(")"):
                    meta_dict[raw_id] = clean_movielens_title_official(raw_title)
                else:
                    meta_dict[raw_id] = raw_title
        return meta_dict
