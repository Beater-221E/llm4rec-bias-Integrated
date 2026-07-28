"""MovieLens adapters package — import registers dataset classes."""

from llm4rec_bias_Integrated.datasets.movielens.ml100k import MovieLens100KAdapter
from llm4rec_bias_Integrated.datasets.movielens.ml1m import MovieLens1MAdapter

__all__ = ["MovieLens100KAdapter", "MovieLens1MAdapter"]
