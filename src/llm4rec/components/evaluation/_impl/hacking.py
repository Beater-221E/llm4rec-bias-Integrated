"""Reward-hacking analysis: training reward vs held-out quality."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from llm4rec.core.reproducibility import write_json


def _zscore(xs: np.ndarray) -> np.ndarray:
    if len(xs) < 2 or float(np.std(xs)) < 1e-12:
        return np.zeros_like(xs, dtype=np.float64)
    return (xs - np.mean(xs)) / np.std(xs)


def _minmax(xs: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(xs)), float(np.max(xs))
    if hi - lo < 1e-12:
        return np.zeros_like(xs, dtype=np.float64)
    return (xs - lo) / (hi - lo)


def normalize_series(values: Sequence[float], method: str = "relative") -> np.ndarray:
    """Normalize a metric trajectory.

    Methods:
      - relative: (x - x0) / max(|x0|, 1.0)  — floor at 1.0 so a zero
        baseline (e.g. pre-RL placeholder) does not explode
      - zscore
      - minmax
    """
    xs = np.asarray(list(values), dtype=np.float64)
    if len(xs) == 0:
        return xs
    if method == "zscore":
        return _zscore(xs)
    if method == "minmax":
        return _minmax(xs)
    # relative to initial
    x0 = xs[0]
    return (xs - x0) / max(abs(x0), 1.0)


def hacking_gap(
    train_rewards: Sequence[float],
    heldout_quality: Sequence[float],
    *,
    method: str = "relative",
) -> dict[str, Any]:
    """Compute hacking gap between reward and held-out quality trajectories.

    ``hacking_gap = Δ_norm(reward) - Δ_norm(heldout)`` using the *final*
    normalized values (change from the first checkpoint).
    """
    if len(train_rewards) != len(heldout_quality):
        raise ValueError("train_rewards and heldout_quality must be aligned")
    if len(train_rewards) < 2:
        return {
            "method": method,
            "hacking_gap": None,
            "delta_reward_raw": None,
            "delta_heldout_raw": None,
            "note": "need ≥2 checkpoints",
        }
    r = np.asarray(train_rewards, dtype=np.float64)
    h = np.asarray(heldout_quality, dtype=np.float64)
    rn = normalize_series(r, method)
    hn = normalize_series(h, method)
    return {
        "method": method,
        "delta_reward_raw": float(r[-1] - r[0]),
        "delta_heldout_raw": float(h[-1] - h[0]),
        "delta_reward_norm": float(rn[-1] - rn[0]),
        "delta_heldout_norm": float(hn[-1] - hn[0]),
        "hacking_gap": float((rn[-1] - rn[0]) - (hn[-1] - hn[0])),
        "reward_series": r.tolist(),
        "heldout_series": h.tolist(),
        "reward_norm_series": rn.tolist(),
        "heldout_norm_series": hn.tolist(),
    }


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def analyze_reward_hacking(
    checkpoints: list[dict[str, Any]],
    *,
    reward_key: str = "train/reward",
    quality_key: str = "eval/hr@10",
    extra_series: dict[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Analyze a list of checkpoint summaries with aligned metrics."""
    rewards = [float(c[reward_key]) for c in checkpoints if reward_key in c]
    quality = [float(c[quality_key]) for c in checkpoints if quality_key in c]
    # Align on intersection order — assume already aligned rows
    n = min(len(rewards), len(quality))
    rewards, quality = rewards[:n], quality[:n]

    report: dict[str, Any] = {
        "n_checkpoints": n,
        "gaps": {
            method: hacking_gap(rewards, quality, method=method)
            for method in ("relative", "zscore", "minmax")
        },
        "correlations": {
            "reward_hr": pearson(rewards, quality),
        },
    }
    if extra_series:
        for name, series in extra_series.items():
            report["correlations"][f"reward_{name}"] = pearson(rewards, list(series)[:n])
    return report


def write_hacking_report(path, report: dict[str, Any]) -> None:
    write_json(path, report)
