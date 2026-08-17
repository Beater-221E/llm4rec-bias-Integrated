"""File-based gather for eval shards.

Independent decode can take hours and finish at different times. NCCL
``all_gather_object`` uses a 10-minute watchdog, so the first finished rank
aborts the job. Shards go to disk; ranks poll for ``.done`` markers instead.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Sequence

from llm4rec.core import distributed as dist_utils
from llm4rec.eval.bias import RankedResult

_LOG = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = float(os.environ.get("LLM4REC_EVAL_GATHER_TIMEOUT", 8 * 3600))
_DEFAULT_POLL_S = float(os.environ.get("LLM4REC_EVAL_GATHER_POLL", 5.0))


def gather_ranked_results(
    local: Sequence[RankedResult],
    shard_dir: Path,
    *,
    name: str,
    timeout_s: float | None = None,
    poll_s: float | None = None,
    rank: int | None = None,
    world_size: int | None = None,
) -> list[RankedResult]:
    """Write this rank's shard, wait for every rank's ``.done``, then load all.

    Single-process returns ``local`` unchanged. Multi-process never touches NCCL.
    """
    r = dist_utils.rank() if rank is None else int(rank)
    w = dist_utils.world_size() if world_size is None else int(world_size)
    rows = list(local)
    if w <= 1:
        return rows

    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = shard_dir / f"{name}.rank{r}.jsonl"
    done_path = shard_dir / f"{name}.rank{r}.done"
    _write_shard(jsonl_path, rows)
    _atomic_write(done_path, f"{len(rows)}\n")
    _LOG.info("[eval] rank %d/%d wrote %d results → %s", r, w, len(rows), jsonl_path)

    done_paths = [shard_dir / f"{name}.rank{i}.done" for i in range(w)]
    _wait_for_done(
        done_paths,
        timeout_s=_DEFAULT_TIMEOUT_S if timeout_s is None else float(timeout_s),
        poll_s=_DEFAULT_POLL_S if poll_s is None else float(poll_s),
    )

    out: list[RankedResult] = []
    for i in range(w):
        out.extend(_read_shard(shard_dir / f"{name}.rank{i}.jsonl"))
    return out


def _write_shard(path: Path, rows: Sequence[RankedResult]) -> None:
    text = "".join(json.dumps(row.to_dict(), ensure_ascii=False) + "\n" for row in rows)
    _atomic_write(path, text)


def _read_shard(path: Path) -> list[RankedResult]:
    if not path.is_file():
        raise FileNotFoundError(f"eval shard missing after .done: {path}")
    out: list[RankedResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(RankedResult.from_dict(json.loads(line)))
    return out


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _wait_for_done(
    done_paths: Sequence[Path],
    *,
    timeout_s: float,
    poll_s: float,
) -> None:
    deadline = time.monotonic() + max(0.0, timeout_s)
    last_log = 0.0
    while True:
        missing = [p.name for p in done_paths if not p.is_file()]
        if not missing:
            return
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(
                f"eval shard gather timed out after {timeout_s:.0f}s; missing {missing}"
            )
        if now - last_log >= 60.0:
            _LOG.info("[eval] waiting for shards: %s", ", ".join(missing))
            last_log = now
        time.sleep(max(0.01, poll_s))
