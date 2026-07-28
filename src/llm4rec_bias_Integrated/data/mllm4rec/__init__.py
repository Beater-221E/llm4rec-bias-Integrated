# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
#
# Original behavior is preserved unless explicitly documented.

"""MLLM4Rec official-compatible data generation pipeline."""

from llm4rec_bias_Integrated.data.mllm4rec.compatibility import (
    load_official_compatible_dataset,
    simulate_official_ranker_prompt,
    simulate_official_retriever_load,
    validate_official_schema,
)
from llm4rec_bias_Integrated.data.mllm4rec.dataset_factory import dataset_factory
from llm4rec_bias_Integrated.data.mllm4rec.schemas import OfficialDatasetDict

__all__ = [
    "OfficialDatasetDict",
    "dataset_factory",
    "load_official_compatible_dataset",
    "simulate_official_ranker_prompt",
    "simulate_official_retriever_load",
    "validate_official_schema",
]
