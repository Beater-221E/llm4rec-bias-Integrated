"""Permutation probe: random candidate-order shuffles (content fixed)."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llm4rec_bias_Integrated.core.schemas import ProbeResult, RecommendationExample
from llm4rec_bias_Integrated.probes.base import BiasProbe
from llm4rec_bias_Integrated.probes.rebuild import rebuild_example
from llm4rec_bias_Integrated.probes.registry import register_probe
from llm4rec_bias_Integrated.probes.scoring import choose_index, score_example


@register_probe("permutation")
class PermutationProbe(BiasProbe):
    name = "permutation"

    def __init__(
        self,
        *,
        n_shuffles: int = 1,
        seed: int = 0,
        **_: Any,
    ) -> None:
        self.n_shuffles = max(1, int(n_shuffles))
        self.seed = int(seed)

    def run(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        examples: list[RecommendationExample],
        *,
        device: Any,
        cfg: dict[str, Any] | None = None,
    ) -> ProbeResult:
        _ = cfg
        rng = random.Random(self.seed)
        agreements: list[float] = []
        hr_orig: list[float] = []
        hr_shuf: list[float] = []

        for ex in examples:
            if ex.candidates is None or ex.target_index is None:
                continue
            orig = score_example(tokenizer, model, device, ex)
            hr_orig.append(orig["hit"])
            titles = list(ex.features["candidate_titles"])
            quants = list(ex.features["pop_quantiles"])
            ids = list(ex.candidates)
            genres = ex.features.get("candidate_genres")
            years = ex.features.get("candidate_years")
            t = int(ex.target_index)
            target_id = ids[t]

            for s in range(self.n_shuffles):
                order = list(range(len(ids)))
                rng.shuffle(order)
                new_ids = [ids[i] for i in order]
                new_titles = [titles[i] for i in order]
                new_quants = [quants[i] for i in order]
                new_genres = (
                    [genres[i] for i in order] if genres is not None else None
                )
                new_years = [years[i] for i in order] if years is not None else None
                new_target = new_ids.index(target_id)
                cf = rebuild_example(
                    ex,
                    candidate_ids=new_ids,
                    candidate_titles=new_titles,
                    pop_quantiles=new_quants,
                    target_index=new_target,
                    candidate_genres=new_genres,
                    candidate_years=new_years,
                    extra_features={"probe_shuffle": s},
                    example_id_suffix=f":perm{s}",
                )
                choice = choose_index(tokenizer, model, device, cf)
                chosen_orig = order[choice]
                agreements.append(float(chosen_orig == orig["choice"]))
                hr_shuf.append(float(choice == new_target))

        metrics = {
            "n": float(len(examples)),
            "agreement_rate": float(np.mean(agreements)) if agreements else 0.0,
            "hr@1_original": float(np.mean(hr_orig)) if hr_orig else 0.0,
            "hr@1_shuffled": float(np.mean(hr_shuf)) if hr_shuf else 0.0,
        }
        metrics["delta_hr@1"] = metrics["hr@1_shuffled"] - metrics["hr@1_original"]
        return ProbeResult(
            name=self.name,
            metrics=metrics,
            details={"n_shuffles": self.n_shuffles},
        )
