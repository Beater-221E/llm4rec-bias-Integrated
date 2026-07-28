# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
#
# Original behavior is preserved unless explicitly documented.

"""Path templates and defaults for MLLM4Rec data pipeline."""

from __future__ import annotations

DEFAULT_MIN_RATING = 0
DEFAULT_MIN_UC = 5
DEFAULT_MIN_SC = 5
DEFAULT_SEED = 42

# Official BLIP2 checkpoint id (used in later phases).
DEFAULT_BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"

TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/original"
TMDB_API_KEY_ENV = "TMDB_API_KEY"

# Official dataset code for ml-latest-small (NOT classic GroupLens ml-100k).
OFFICIAL_ML100K_CODE = "ml-100k"
ML_LATEST_SMALL_URL = (
    "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
)

CLASSIC_ML100K_CODE = "ml-100k-classic"
CLASSIC_ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

ML1M_CODE = "ml-1m"
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

DATASET_PKL_NAME = "dataset.pkl"
IMG_DIR_NAME = "img"

PREPROCESSED_FOLDER_TEMPLATE = (
    "{code}_min_rating{min_rating}-min_uc{min_uc}-min_sc{min_sc}"
)


def preprocessed_folder_name(
    *,
    code: str,
    min_rating: int,
    min_uc: int,
    min_sc: int,
) -> str:
    """Match MLLM4Rec AbstractDataset._get_preprocessed_folder_path naming."""
    return PREPROCESSED_FOLDER_TEMPLATE.format(
        code=code,
        min_rating=min_rating,
        min_uc=min_uc,
        min_sc=min_sc,
    )
