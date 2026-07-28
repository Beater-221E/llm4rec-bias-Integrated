# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
#
# Original behavior is preserved unless explicitly documented.

"""MLLM4Rec official-compatible data generation pipeline."""

from llm4rec.workflows.mllm4rec.data.compatibility import (
    load_official_compatible_dataset,
    simulate_official_ranker_prompt,
    simulate_official_retriever_load,
    validate_official_schema,
)
from llm4rec.workflows.mllm4rec.data.dataset_factory import dataset_factory
from llm4rec.workflows.mllm4rec.data.schemas import OfficialDatasetDict

__all__ = [
    "OfficialDatasetDict",
    "dataset_factory",
    "load_official_compatible_dataset",
    "simulate_official_ranker_prompt",
    "simulate_official_retriever_load",
    "validate_official_schema",
]
