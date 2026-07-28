# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
#
# Original behavior is preserved unless explicitly documented.

"""Dataset registry / factory."""

from __future__ import annotations

from typing import Type

from llm4rec_bias_Integrated.data.mllm4rec.base_dataset import BaseMovieLensDataset
from llm4rec_bias_Integrated.data.mllm4rec.config import MLLM4RecDataConfig
from llm4rec_bias_Integrated.data.mllm4rec.movielens_100k import ML100KClassicDataset, ML100KDataset
from llm4rec_bias_Integrated.data.mllm4rec.movielens_1m import ML1MDataset

DATASETS: dict[str, Type[BaseMovieLensDataset]] = {
    ML100KDataset.code(): ML100KDataset,
    ML100KClassicDataset.code(): ML100KClassicDataset,
    ML1MDataset.code(): ML1MDataset,
}


def dataset_factory(cfg: MLLM4RecDataConfig) -> BaseMovieLensDataset:
    """Construct a dataset instance from resolved config (official factory shape)."""
    try:
        cls = DATASETS[cfg.dataset_code]
    except KeyError as exc:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(
            f"Unknown dataset_code={cfg.dataset_code!r}. Known: {known}"
        ) from exc
    return cls(cfg)
