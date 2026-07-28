"""Evaluation interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm4rec_bias_Integrated.core.schemas import EvaluationResult


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        model,
        dataset,
        split: str,
    ) -> EvaluationResult:
        ...
