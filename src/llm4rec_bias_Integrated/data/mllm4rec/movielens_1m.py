# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
#
# MovieLens-1M is an MLLM4Rec-compatible *extension* (not an official experiment set).

"""MovieLens-1M dataset (ratings.dat / movies.dat)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from llm4rec_bias_Integrated.data.mllm4rec._download import download_and_extract_zip_folder
from llm4rec_bias_Integrated.data.mllm4rec.base_dataset import BaseMovieLensDataset
from llm4rec_bias_Integrated.data.mllm4rec.constants import ML1M_CODE, ML1M_URL
from llm4rec_bias_Integrated.data.mllm4rec.movielens_100k import clean_movielens_title_official

logger = logging.getLogger("llm4rec_bias_Integrated.mllm4rec")


class ML1MDataset(BaseMovieLensDataset):
    """MLLM4Rec-compatible extension for MovieLens-1M."""

    @classmethod
    def code(cls) -> str:
        return ML1M_CODE

    @classmethod
    def url(cls) -> str:
        return ML1M_URL

    @classmethod
    def all_raw_file_names(cls) -> list[str]:
        return ["ratings.dat", "movies.dat", "users.dat"]

    def maybe_download_raw_dataset(self) -> None:
        folder_path = self.raw_folder_path()
        if folder_path.is_dir() and (folder_path / "ratings.dat").is_file():
            logger.info("Raw ml-1m already exists. Skip downloading")
            return
        parent = folder_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        download_and_extract_zip_folder(
            self.url(),
            parent / "ml-1m",
            zip_content_is_folder=True,
        )

    def load_ratings_df(self) -> pd.DataFrame:
        file_path = self.raw_folder_path() / "ratings.dat"
        df = pd.read_csv(
            file_path,
            sep="::",
            engine="python",
            header=None,
            names=["uid", "sid", "rating", "timestamp"],
        )
        return df

    def load_meta_dict(self) -> dict[Any, str]:
        file_path = self.raw_folder_path() / "movies.dat"
        meta_dict: dict[Any, str] = {}
        with file_path.open(encoding="ISO-8859-1") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("::")
                if len(parts) < 2:
                    continue
                raw_id = int(parts[0])
                raw_title = parts[1]
                if len(raw_title) >= 7 and raw_title[-6:-1].isdigit() and raw_title.endswith(")"):
                    meta_dict[raw_id] = clean_movielens_title_official(raw_title)
                else:
                    meta_dict[raw_id] = raw_title
        return meta_dict
