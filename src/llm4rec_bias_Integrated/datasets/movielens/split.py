"""Train / validation / test splitters and leakage checks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from llm4rec_bias_Integrated.core.exceptions import DatasetValidationError
from llm4rec_bias_Integrated.core.schemas import DatasetSplits, Interaction
from llm4rec_bias_Integrated.datasets.movielens.common import chronological_sequences

SplitMethod = Literal["leave_one_out", "chronological_ratio", "fixed"]


def leave_one_out_split(
    interactions: list[Interaction],
    *,
    min_user_interactions: int = 5,
) -> DatasetSplits:
    """Per-user LOO: last → test, second-last → val, earlier → train."""
    sequences = chronological_sequences(interactions)
    train: list[Interaction] = []
    validation: list[Interaction] = []
    test: list[Interaction] = []
    kept_users = 0
    for _user, seq in sequences.items():
        if len(seq) < min_user_interactions:
            continue
        kept_users += 1
        test.append(seq[-1])
        validation.append(seq[-2])
        train.extend(seq[:-2])
    return DatasetSplits(
        train=train,
        validation=validation,
        test=test,
        metadata={
            "method": "leave_one_out",
            "min_user_interactions": min_user_interactions,
            "n_users": kept_users,
        },
    )


def chronological_ratio_split(
    interactions: list[Interaction],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    min_user_interactions: int = 5,
) -> DatasetSplits:
    """Per-user chronological cut by ratios (remainder → test)."""
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1.0:
        raise DatasetValidationError(
            "chronological_ratio requires train_ratio>0, val_ratio>=0, "
            "and train_ratio+val_ratio < 1"
        )
    sequences = chronological_sequences(interactions)
    train: list[Interaction] = []
    validation: list[Interaction] = []
    test: list[Interaction] = []
    kept = 0
    for _user, seq in sequences.items():
        if len(seq) < min_user_interactions:
            continue
        kept += 1
        n = len(seq)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        # Ensure at least one test when possible
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        train.extend(seq[:n_train])
        validation.extend(seq[n_train : n_train + n_val])
        test.extend(seq[n_train + n_val :])
    return DatasetSplits(
        train=train,
        validation=validation,
        test=test,
        metadata={
            "method": "chronological_ratio",
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "min_user_interactions": min_user_interactions,
            "n_users": kept,
        },
    )


def validate_no_leakage(splits: DatasetSplits) -> None:
    """Fail if the same (user, item, timestamp) appears across splits."""

    def _keys(rows: list[Interaction]) -> set[tuple[str, str, int]]:
        return {(r.user_id, r.item_id, r.timestamp) for r in rows}

    train_k, val_k, test_k = _keys(splits.train), _keys(splits.validation), _keys(splits.test)
    if train_k & val_k:
        raise DatasetValidationError("Leakage: train ∩ validation is non-empty")
    if train_k & test_k:
        raise DatasetValidationError("Leakage: train ∩ test is non-empty")
    if val_k & test_k:
        raise DatasetValidationError("Leakage: validation ∩ test is non-empty")


def validate_example_integrity(
    history_item_ids: list[str],
    target_item_id: str,
    candidates: list[str] | None,
) -> None:
    """History must not contain target; candidates must include target exactly once."""
    if target_item_id in history_item_ids:
        raise DatasetValidationError(
            f"History contains target item {target_item_id}"
        )
    if candidates is not None:
        if target_item_id not in candidates:
            raise DatasetValidationError(
                f"Candidates missing target item {target_item_id}"
            )
        if candidates.count(target_item_id) != 1:
            raise DatasetValidationError(
                f"Target {target_item_id} appears {candidates.count(target_item_id)} times"
            )


def user_item_sets(splits: DatasetSplits) -> dict[str, Any]:
    def items(rows: list[Interaction]) -> set[str]:
        return {r.item_id for r in rows}

    def users(rows: list[Interaction]) -> set[str]:
        return {r.user_id for r in rows}

    return {
        "n_train": len(splits.train),
        "n_validation": len(splits.validation),
        "n_test": len(splits.test),
        "n_users_train": len(users(splits.train)),
        "n_users_validation": len(users(splits.validation)),
        "n_users_test": len(users(splits.test)),
        "n_items_train": len(items(splits.train)),
        "n_items_validation": len(items(splits.validation)),
        "n_items_test": len(items(splits.test)),
    }


def per_user_targets(
    sequences: dict[str, list[Interaction]],
    split: str,
    *,
    min_user_interactions: int = 5,
) -> list[tuple[str, list[Interaction], Interaction]]:
    """Return (user_id, history_interactions, target_interaction) for a split.

    For leave-one-out semantics:
      train  → every prefix position before the last two items
      validation → history = seq[:-2], target = seq[-2]
      test → history = seq[:-1], target = seq[-1]
    """
    out: list[tuple[str, list[Interaction], Interaction]] = []
    for user_id, seq in sequences.items():
        if len(seq) < min_user_interactions:
            continue
        if split == "test":
            out.append((user_id, seq[:-1], seq[-1]))
        elif split in {"validation", "val"}:
            out.append((user_id, seq[:-2], seq[-2]))
        elif split == "train":
            # All positions with at least 2 history items and before val/test
            for t in range(2, len(seq) - 2):
                out.append((user_id, seq[:t], seq[t]))
        else:
            raise DatasetValidationError(f"Unknown split '{split}'")
    return out
