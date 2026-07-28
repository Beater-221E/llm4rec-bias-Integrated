"""Shared MovieLens adapter logic (download cache, splits, examples)."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from llm4rec_bias_Integrated.core.exceptions import DatasetValidationError
from llm4rec_bias_Integrated.core.reproducibility import fingerprint_payload, write_json
from llm4rec_bias_Integrated.core.schemas import (
    DatasetSplits,
    Interaction,
    RecommendationExample,
    TaskSpec,
)
from llm4rec_bias_Integrated.datasets.base import DatasetAdapter
from llm4rec_bias_Integrated.datasets.movielens.common import (
    ItemMetadata,
    chronological_sequences,
    filter_sequences_by_length,
    popularity_from_train_region,
    popularity_summary,
)
from llm4rec_bias_Integrated.datasets.movielens.split import (
    chronological_ratio_split,
    leave_one_out_split,
    per_user_targets,
    user_item_sets,
    validate_example_integrity,
    validate_no_leakage,
)
from llm4rec_bias_Integrated.datasets.sampling.base import get_sampler
from llm4rec_bias_Integrated.datasets.transforms.candidates import build_candidate_list
from llm4rec_bias_Integrated.datasets.transforms.history import history_item_ids, truncate_history
from llm4rec_bias_Integrated.prompts.candidate_choice import LETTERS, build_candidate_choice_messages

logger = logging.getLogger("llm4rec_bias_Integrated")


class MovieLensAdapterBase(DatasetAdapter):
    """Common pipeline for ML-100K / ML-1M."""

    name: str = "movielens"
    dataset_slug: str = "movielens"

    def __init__(
        self,
        *,
        data_root: Path,
        rating_threshold: float = 4.0,
        split: str = "leave_one_out",
        history_max_length: int = 20,
        candidate_size: int = 10,
        negative_sampling: str = "uniform",
        target_position: str = "random",
        framing: str = "neutral",
        min_user_interactions: int = 5,
        seed: int = 42,
        train_limit: int | None = None,
        eval_limit: int | None = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        force_download: bool = False,
        **_: Any,
    ) -> None:
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw" / self.dataset_slug
        self.processed_dir = self.data_root / "processed" / self.dataset_slug
        self.cache_dir = self.data_root / "cache" / self.dataset_slug
        self.rating_threshold = float(rating_threshold)
        self.split_method = split
        self.history_max_length = int(history_max_length)
        self.candidate_size = int(candidate_size)
        self.negative_sampling = negative_sampling
        self.target_position = target_position
        self.framing = framing
        self.min_user_interactions = int(min_user_interactions)
        self.seed = int(seed)
        self.train_limit = train_limit
        self.eval_limit = eval_limit
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.force_download = force_download

        self._interactions: list[Interaction] | None = None
        self._item_meta: dict[str, ItemMetadata] | None = None
        self._splits: DatasetSplits | None = None
        self._sequences: dict[str, list[Interaction]] | None = None
        self._counts: dict[str, int] | None = None
        self._quantiles: dict[str, float] | None = None

    # --- subclass hooks -------------------------------------------------
    def _ensure_raw(self) -> Path:
        raise NotImplementedError

    def _parse_raw(self, raw_path: Path) -> tuple[list[Interaction], dict[str, ItemMetadata]]:
        raise NotImplementedError

    # --- DatasetAdapter -------------------------------------------------
    def download(self) -> None:
        path = self._ensure_raw()
        logger.info("Raw data ready at %s", path)

    def preprocess(self) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        raw = self._ensure_raw()
        interactions, meta = self._parse_raw(raw)
        # Keep only ratings at/above threshold (already filtered in parsers usually)
        interactions = [
            ix
            for ix in interactions
            if ix.rating is None or ix.rating >= self.rating_threshold
        ]
        if not interactions:
            raise DatasetValidationError(
                f"No interactions remain after rating_threshold={self.rating_threshold}"
            )
        self._interactions = interactions
        self._item_meta = meta
        sequences = chronological_sequences(interactions)
        sequences = filter_sequences_by_length(sequences, self.min_user_interactions)
        if not sequences:
            raise DatasetValidationError(
                f"No users with >= {self.min_user_interactions} interactions"
            )
        self._sequences = {
            user: seq
            for user, seq in sorted(sequences.items(), key=lambda kv: kv[0])
        }
        # Flatten filtered interactions in stable order
        flat = [ix for _user, seq in self._sequences.items() for ix in seq]
        self._interactions = flat
        self._counts, self._quantiles = popularity_from_train_region(
            self._sequences, holdout=2
        )
        self._splits = self._make_splits(flat)
        validate_no_leakage(self._splits)
        self._write_cache()

    def load_interactions(self) -> list[Interaction]:
        self._ensure_processed()
        assert self._interactions is not None
        return list(self._interactions)

    def build_splits(self) -> DatasetSplits:
        self._ensure_processed()
        assert self._splits is not None
        return self._splits

    def build_examples(
        self,
        split: str,
        task_spec: TaskSpec,
    ) -> list[RecommendationExample]:
        self._ensure_processed()
        assert self._sequences is not None
        assert self._item_meta is not None
        assert self._counts is not None
        assert self._quantiles is not None

        if task_spec.task not in {"candidate_choice", "grpo4rec", "mllm4rec"}:
            # SID tasks will plug in later; still allow candidate_choice alias
            if task_spec.task != "candidate_choice":
                # default path for Phase 2 is candidate choice
                pass

        rng = random.Random(self.seed + hash(split) % 10_000)
        targets = per_user_targets(
            self._sequences,
            split,
            min_user_interactions=self.min_user_interactions,
        )

        pool_items = sorted(self._counts.keys())
        sampler_kwargs: dict[str, Any] = {"item_ids": pool_items}
        if task_spec.negative_sampling in {"popularity", "pop", "hard_negative"}:
            sampler_kwargs["counts"] = self._counts
        sampler = get_sampler(task_spec.negative_sampling, **sampler_kwargs)

        examples: list[RecommendationExample] = []
        for user_id, hist_ix, target_ix in targets:
            hist_ix = truncate_history(hist_ix, task_spec.history_max_length)
            hist_ids = history_item_ids(hist_ix)
            if len(hist_ids) < 2:
                continue
            target_id = target_ix.item_id
            if target_id not in self._item_meta:
                continue

            if task_spec.negative_sampling == "exposure_matched":
                sampler = get_sampler(
                    "exposure_matched",
                    item_ids=pool_items,
                    quantiles=self._quantiles,
                    target_quantile=self._quantiles.get(target_id, 0.5),
                )

            exclude = set(hist_ids)
            candidates, target_index = build_candidate_list(
                target_item_id=target_id,
                exclude=exclude,
                sampler=sampler,
                candidate_size=task_spec.candidate_size,
                target_position=task_spec.target_position,
                rng=rng,
            )
            validate_example_integrity(hist_ids, target_id, candidates)

            hist_titles = [self._item_meta[i].title for i in hist_ids if i in self._item_meta]
            cand_titles = [self._item_meta[i].title for i in candidates]
            cand_quants = [self._quantiles.get(i, 0.5) for i in candidates]
            cand_genres = [self._item_meta[i].genres for i in candidates]
            cand_years = [self._item_meta[i].release_year for i in candidates]
            messages = build_candidate_choice_messages(
                hist_titles,
                cand_titles,
                cand_quants,
                task_spec.framing,
                candidate_genres=cand_genres,
                candidate_years=cand_years,
            )
            example = RecommendationExample(
                example_id=f"{self.name}:{split}:{user_id}:{target_id}:{target_index}",
                user_id=user_id,
                history_item_ids=hist_ids,
                target_item_id=target_id,
                candidates=candidates,
                prompt_messages=messages,
                target_text=LETTERS[target_index],
                target_index=target_index,
                features={
                    "history_titles": hist_titles,
                    "candidate_titles": cand_titles,
                    "pop_quantiles": cand_quants,
                    "candidate_genres": [list(g) for g in cand_genres],
                    "candidate_years": list(cand_years),
                    "framing": task_spec.framing,
                    "negative_sampling": task_spec.negative_sampling,
                    "target_position_policy": task_spec.target_position,
                    "item_popularity": self._counts.get(target_id, 0),
                    "popularity_quantile": self._quantiles.get(target_id, 0.5),
                    "history_popularity_mean": float(
                        sum(self._quantiles.get(i, 0.5) for i in hist_ids) / len(hist_ids)
                    ),
                    "candidate_positions": list(range(len(candidates))),
                    "genres": list(self._item_meta[target_id].genres),
                    "release_year": self._item_meta[target_id].release_year,
                },
            )
            examples.append(example)

        # Deterministic shuffle before optional limits so caps are not user-prefix biased.
        rng.shuffle(examples)
        limit = self.train_limit if split == "train" else self.eval_limit
        if limit is not None:
            examples = examples[: int(limit)]
        return examples

    def fingerprint(self) -> str:
        self._ensure_processed()
        assert self._interactions is not None
        assert self._splits is not None
        payload = {
            "dataset": self.name,
            "rating_threshold": self.rating_threshold,
            "split": self.split_method,
            "min_user_interactions": self.min_user_interactions,
            "seed": self.seed,
            "n_interactions": len(self._interactions),
            "n_train": len(self._splits.train),
            "n_val": len(self._splits.validation),
            "n_test": len(self._splits.test),
            "first_user": self._interactions[0].user_id if self._interactions else None,
            "first_item": self._interactions[0].item_id if self._interactions else None,
            "item_id_sample": sorted({ix.item_id for ix in self._interactions[:200]}),
        }
        return fingerprint_payload(payload)

    def summary(self) -> dict[str, Any]:
        self._ensure_processed()
        assert self._interactions is not None
        assert self._item_meta is not None
        assert self._splits is not None
        assert self._counts is not None
        assert self._quantiles is not None
        assert self._sequences is not None
        return {
            "name": self.name,
            "n_interactions": len(self._interactions),
            "n_users": len(self._sequences),
            "n_items_catalog": len(self._item_meta),
            "n_items_with_train_pop": len(self._counts),
            "split_sizes": user_item_sets(self._splits),
            "popularity": popularity_summary(self._counts, self._quantiles),
            "fingerprint": self.fingerprint(),
            "first_interaction": {
                "user_id": self._interactions[0].user_id,
                "item_id": self._interactions[0].item_id,
                "rating": self._interactions[0].rating,
                "timestamp": self._interactions[0].timestamp,
            },
            "rating_threshold": self.rating_threshold,
            "split_method": self.split_method,
            "seed": self.seed,
        }

    def item_metadata(self) -> dict[str, ItemMetadata]:
        self._ensure_processed()
        assert self._item_meta is not None
        return dict(self._item_meta)

    # --- internals ------------------------------------------------------
    def _make_splits(self, interactions: list[Interaction]) -> DatasetSplits:
        if self.split_method == "leave_one_out":
            return leave_one_out_split(
                interactions,
                min_user_interactions=self.min_user_interactions,
            )
        if self.split_method == "chronological_ratio":
            return chronological_ratio_split(
                interactions,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                min_user_interactions=self.min_user_interactions,
            )
        if self.split_method == "fixed":
            raise DatasetValidationError(
                "split=fixed requires a pre-generated split file (not yet configured)"
            )
        raise DatasetValidationError(f"Unknown split method '{self.split_method}'")

    def _ensure_processed(self) -> None:
        if self._interactions is None or self._splits is None:
            cache = self.processed_dir / "manifest.json"
            if cache.exists() and self._try_load_cache():
                return
            self.preprocess()

    def _write_cache(self) -> None:
        assert self._interactions is not None
        assert self._item_meta is not None
        assert self._splits is not None
        assert self._counts is not None
        assert self._quantiles is not None
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        interactions_path = self.processed_dir / "interactions.jsonl"
        with interactions_path.open("w", encoding="utf-8") as fh:
            for ix in self._interactions:
                fh.write(
                    json.dumps(
                        {
                            "user_id": ix.user_id,
                            "item_id": ix.item_id,
                            "rating": ix.rating,
                            "timestamp": ix.timestamp,
                            "metadata": ix.metadata,
                        }
                    )
                    + "\n"
                )
        meta_path = self.processed_dir / "item_meta.json"
        write_json(
            meta_path,
            {
                iid: {
                    "title": m.title,
                    "genres": list(m.genres),
                    "release_year": m.release_year,
                }
                for iid, m in self._item_meta.items()
            },
        )
        write_json(
            self.processed_dir / "popularity.json",
            {"counts": self._counts, "quantiles": self._quantiles},
        )
        write_json(
            self.processed_dir / "manifest.json",
            {
                "dataset": self.name,
                "fingerprint": self.fingerprint(),
                "rating_threshold": self.rating_threshold,
                "split": self.split_method,
                "seed": self.seed,
                "min_user_interactions": self.min_user_interactions,
                "n_interactions": len(self._interactions),
                "n_train": len(self._splits.train),
                "n_validation": len(self._splits.validation),
                "n_test": len(self._splits.test),
            },
        )
        # Persist split interaction keys for reload
        for split_name, rows in (
            ("train", self._splits.train),
            ("validation", self._splits.validation),
            ("test", self._splits.test),
        ):
            path = self.processed_dir / f"split_{split_name}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for ix in rows:
                    fh.write(
                        json.dumps(
                            {
                                "user_id": ix.user_id,
                                "item_id": ix.item_id,
                                "rating": ix.rating,
                                "timestamp": ix.timestamp,
                            }
                        )
                        + "\n"
                    )

    def _try_load_cache(self) -> bool:
        manifest_path = self.processed_dir / "manifest.json"
        if not manifest_path.exists():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("rating_threshold") != self.rating_threshold
            or manifest.get("split") != self.split_method
            or manifest.get("seed") != self.seed
            or manifest.get("min_user_interactions") != self.min_user_interactions
        ):
            return False
        interactions: list[Interaction] = []
        with (self.processed_dir / "interactions.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                interactions.append(
                    Interaction(
                        user_id=str(row["user_id"]),
                        item_id=str(row["item_id"]),
                        rating=row.get("rating"),
                        timestamp=int(row["timestamp"]),
                        metadata=row.get("metadata") or {},
                    )
                )
        meta_raw = json.loads((self.processed_dir / "item_meta.json").read_text(encoding="utf-8"))
        item_meta = {
            iid: ItemMetadata(
                item_id=iid,
                title=payload["title"],
                genres=tuple(payload.get("genres") or ()),
                release_year=payload.get("release_year"),
            )
            for iid, payload in meta_raw.items()
        }
        pop = json.loads((self.processed_dir / "popularity.json").read_text(encoding="utf-8"))
        counts = {str(k): int(v) for k, v in pop["counts"].items()}
        quantiles = {str(k): float(v) for k, v in pop["quantiles"].items()}

        def _load_split(name: str) -> list[Interaction]:
            rows: list[Interaction] = []
            path = self.processed_dir / f"split_{name}.jsonl"
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    rows.append(
                        Interaction(
                            user_id=str(row["user_id"]),
                            item_id=str(row["item_id"]),
                            rating=row.get("rating"),
                            timestamp=int(row["timestamp"]),
                        )
                    )
            return rows

        splits = DatasetSplits(
            train=_load_split("train"),
            validation=_load_split("validation"),
            test=_load_split("test"),
            metadata={"method": self.split_method, "from_cache": True},
        )
        self._interactions = interactions
        self._item_meta = item_meta
        self._counts = counts
        self._quantiles = quantiles
        self._splits = splits
        self._sequences = chronological_sequences(interactions)
        return True
