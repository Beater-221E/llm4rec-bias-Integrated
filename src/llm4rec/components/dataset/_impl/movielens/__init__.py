"""MovieLens adapters package — import registers dataset classes."""

from llm4rec.components.dataset._impl.movielens.ml100k import MovieLens100KAdapter
from llm4rec.components.dataset._impl.movielens.ml1m import MovieLens1MAdapter

__all__ = ["MovieLens100KAdapter", "MovieLens1MAdapter"]
