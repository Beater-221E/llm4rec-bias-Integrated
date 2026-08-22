"""Periodic mid-training checkpoint helpers (SFT / RL / DPO / distill).

Config (``checkpoint`` in YAML):

* ``save_steps``: int step interval, float in ``(0, 1)`` as a fraction of
  ``max_steps``, or ``null``/``0``/``false`` to disable mid-saves.
* ``save_total_limit``: keep at most this many ``checkpoint-{step}`` dirs
  (oldest deleted). Stage ``final/`` and ``best/`` are never pruned here.
* ``save_best``: when eval improves, overwrite ``best/`` (default true).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from llm4rec.core import distributed as dist_utils

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def resolve_save_steps(
    cfg: dict[str, Any],
    stage_cfg: dict[str, Any] | None = None,
    *,
    max_steps: int | None = None,
    as_int: bool = False,
) -> int | float | None:
    """Resolve ``save_steps`` from stage override or global ``checkpoint``.

    When ``as_int`` is True (custom loops like GRPO/DPO), float ratios are
    converted with ``max_steps``. HF Trainer can consume the float ratio itself.
    """
    ckpt = cfg.get("checkpoint") or {}
    stage = stage_cfg or {}
    value = stage.get("save_steps", ckpt.get("save_steps"))
    if value in (None, 0, "null", False):
        return None
    if isinstance(value, float) and 0.0 < value < 1.0:
        if as_int:
            if max_steps is None or max_steps <= 0:
                return None
            return max(1, int(max_steps * value))
        return value
    interval = int(value)
    if interval <= 0:
        return None
    return interval


def resolve_save_total_limit(cfg: dict[str, Any]) -> int:
    return max(1, int((cfg.get("checkpoint") or {}).get("save_total_limit") or 1))


def save_best_enabled(cfg: dict[str, Any] | None) -> bool:
    ckpt = (cfg or {}).get("checkpoint") or {}
    return bool(ckpt.get("save_best", True))


def should_save_at_step(step: int, interval: int | None) -> bool:
    if interval is None or interval <= 0 or step <= 0:
        return False
    return step % int(interval) == 0


def list_step_checkpoints(output_dir: Path | str) -> list[tuple[int, Path]]:
    root = Path(output_dir)
    if not root.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = _CKPT_RE.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort(key=lambda x: x[0])
    return found


def prune_step_checkpoints(output_dir: Path | str, save_total_limit: int) -> list[Path]:
    """Delete oldest ``checkpoint-*`` dirs beyond ``save_total_limit``. Rank0 only."""
    limit = max(1, int(save_total_limit))
    kept = list_step_checkpoints(output_dir)
    removed: list[Path] = []
    while len(kept) > limit:
        _, path = kept.pop(0)
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path)
    return removed


def save_step_checkpoint(
    model: Any,
    output_dir: Path | str,
    step: int,
    *,
    tokenizer: Any = None,
    cfg: dict[str, Any] | None = None,
    logger: Any = None,
    tag: str = "train",
) -> Path:
    """Write ``output_dir/checkpoint-{step}`` (all ranks sync), then prune.

    Safe under DDP/FSDP: uses ``save_pretrained_distributed`` + barrier.
    """
    out = Path(output_dir)
    ckpt_dir = out / f"checkpoint-{step}"
    dist_utils.save_pretrained_distributed(
        model, ckpt_dir, tokenizer=tokenizer, is_main=dist_utils.is_main()
    )
    if dist_utils.is_main():
        limit = resolve_save_total_limit(cfg or {})
        removed = prune_step_checkpoints(out, limit)
        log = getattr(logger, "info", None) if logger is not None else None
        if callable(log):
            extra = f"，清理 {len(removed)} 个旧 checkpoint" if removed else ""
            log(f"[{tag}] mid-checkpoint → {ckpt_dir}{extra}")
    dist_utils.barrier(f"{tag}_ckpt_{step}")
    return ckpt_dir


def save_best_checkpoint(
    model: Any,
    output_dir: Path | str,
    *,
    metric: float,
    step: int,
    tokenizer: Any = None,
    logger: Any = None,
    tag: str = "train",
    metric_name: str = "loss",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Overwrite ``output_dir/best`` with the current weights (all ranks sync)."""
    out = Path(output_dir)
    best_dir = out / "best"
    dist_utils.save_pretrained_distributed(
        model, best_dir, tokenizer=tokenizer, is_main=dist_utils.is_main()
    )
    if dist_utils.is_main():
        payload = {
            "step": int(step),
            "metric": float(metric),
            "metric_name": metric_name,
        }
        if extra:
            payload.update(extra)
        (best_dir / "best_metric.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        log = getattr(logger, "info", None) if logger is not None else None
        if callable(log):
            log(f"[{tag}] best checkpoint → {best_dir} ({metric_name}={metric:.4f} @ step {step})")
    dist_utils.barrier(f"{tag}_best_{step}")
    return best_dir


def is_better_metric(
    metric: float,
    best: float | None,
    *,
    lower_is_better: bool = True,
) -> bool:
    if best is None:
        return True
    return float(metric) < float(best) if lower_is_better else float(metric) > float(best)
