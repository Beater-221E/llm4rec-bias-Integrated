"""Candidate letter log-prob ranking evaluator (GPU only).

Uses upstream-compatible metric aggregation from
``llm4rec.compatibility.llm4rec_bias_eval``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llm4rec.compatibility.llm4rec_bias_eval import ndcg_at as upstream_ndcg_at
from llm4rec.core.exceptions import ConfigurationError
from llm4rec.core.schemas import EvaluationResult, RecommendationExample
from llm4rec.components.evaluation._impl.adapter import examples_to_ranked_letter, to_evaluation_result
from llm4rec.components.evaluation._impl.base import Evaluator
from llm4rec.compatibility.llm4rec_bias_eval import aggregate_letter_route
from llm4rec.components.model._impl.base import require_cuda
from llm4rec.components.prompts.candidate_choice import LETTERS


def ndcg_at(rank: int, k: int = 5) -> float:
    """Alias kept for callers; delegates to upstream-compatible definition."""
    return upstream_ndcg_at(rank, k)


@torch.no_grad()
def score_letters(
    tok: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
    messages: list[dict[str, str]],
    n: int,
) -> np.ndarray:
    """Log-prob of each letter A..<n> as the next assistant token."""
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    ids = enc["input_ids"] if not isinstance(enc, torch.Tensor) else enc
    out = model(ids.to(device))
    logits = out.logits[0, -1]
    logp = torch.log_softmax(logits.float(), dim=-1)
    scores = []
    for i in range(n):
        letter_ids = tok.encode(LETTERS[i], add_special_tokens=False)
        scores.append(logp[letter_ids[0]].item())
    return np.asarray(scores, dtype=np.float64)


class CandidateLogProbEvaluator(Evaluator):
    """Rank candidates by next-token letter log-prob (not free-text generation)."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        *,
        top_k: list[int] | None = None,
        device: str | torch.device | None = None,
        predictions_dir: Path | None = None,
    ) -> None:
        require_cuda()
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.top_k = sorted(top_k or [1, 5, 10])
        self.device = torch.device(device or "cuda")
        self.model.to(self.device)
        self.predictions_dir = predictions_dir

    def evaluate(
        self,
        model=None,
        dataset: list[RecommendationExample] | None = None,
        split: str = "test",
    ) -> EvaluationResult:
        require_cuda()
        model = model or self.model
        if dataset is None:
            raise ConfigurationError("dataset (list of RecommendationExample) is required")
        model.eval()
        model.to(self.device)

        ranks: list[int] = []
        chosen: list[int] = []
        predictions: list[dict[str, Any]] = []

        for ex in dataset:
            n = len(ex.candidates or [])
            if n == 0 or ex.target_index is None:
                raise ConfigurationError(
                    f"Example {ex.example_id} missing candidates/target_index"
                )
            scores = score_letters(
                self.tokenizer, model, self.device, ex.prompt_messages, n
            )
            order = np.argsort(-scores)
            rank = int(np.where(order == ex.target_index)[0][0])
            choice = int(order[0])
            ranks.append(rank)
            chosen.append(choice)
            predictions.append(
                {
                    "example_id": ex.example_id,
                    "user_id": ex.user_id,
                    "target_index": ex.target_index,
                    "chosen_index": choice,
                    "rank": rank,
                    "scores": scores.tolist(),
                    "order": order.tolist(),
                    "pop_quantiles": ex.features.get("pop_quantiles"),
                }
            )

        ranked = examples_to_ranked_letter(dataset, ranks, chosen)
        compat = aggregate_letter_route(ranked, top_k=self.top_k)

        pred_path = None
        if self.predictions_dir is not None:
            self.predictions_dir.mkdir(parents=True, exist_ok=True)
            pred_path = self.predictions_dir / f"predictions_{split}.jsonl"
            with pred_path.open("w", encoding="utf-8") as fh:
                for row in predictions:
                    fh.write(json.dumps(row) + "\n")

        return to_evaluation_result(
            compat,
            split=split,
            predictions_path=pred_path,
            extra_metadata={"route": "letter", "scoring": "letter_logprob"},
        )
