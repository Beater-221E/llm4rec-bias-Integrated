"""Rank-0 progress: tqdm only on a real TTY; otherwise a sparse INFO line.

``run.sh | tee`` makes stderr a pipe. tqdm ``\\r`` then becomes a new line
every update and floods the terminal. Under a pipe we never draw a bar —
rank 0 logs one aggregated line every few seconds instead.

Multi-GPU eval: ranks publish local counts to files; rank 0 sums them.
No NCCL on the hot path.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

_LOG = logging.getLogger(__name__)
_BAR_FORMAT = "{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def _atomic_write_int(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(str(int(value)), encoding="utf-8")
    tmp.replace(path)


def _read_int(path: Path) -> int:
    try:
        return max(0, int(path.read_text(encoding="utf-8").strip() or 0))
    except (OSError, ValueError):
        return 0


class ProgressHandle:
    """All ranks call ``update``; only rank 0 reports aggregated progress."""

    def __init__(
        self,
        *,
        bar: tqdm | None,
        local_total: int,
        global_total: int,
        rank: int,
        world: int,
        paths: list[Path] | None,
        desc: str = "eval",
        log_interval_s: float = 5.0,
    ) -> None:
        self.bar = bar
        self.local_done = 0
        self.local_total = int(local_total)
        self.global_total = int(global_total)
        self.rank = int(rank)
        self.world = int(world)
        self.paths = paths
        self.desc = str(desc or "eval")
        self.log_interval_s = max(0.5, float(log_interval_s))
        self._t0 = time.monotonic()
        self._last_log = 0.0
        if self.paths:
            _atomic_write_int(self.paths[self.rank], 0)
        if self.rank == 0 and self.global_total > 0:
            self._log(0)

    def update(self, n: int = 1) -> int:
        self.local_done += int(n)
        if self.paths:
            _atomic_write_int(self.paths[self.rank], self.local_done)
        done = self.global_done()
        if self.bar is not None:
            self.bar.n = min(done, self.bar.total or done)
            self.bar.refresh()
        if self.rank == 0:
            self._maybe_log(done)
        return done

    def global_done(self) -> int:
        if not self.paths:
            return self.local_done
        return sum(_read_int(p) for p in self.paths)

    def _maybe_log(self, done: int) -> None:
        total = self.global_total
        if total <= 0:
            return
        now = time.monotonic()
        if done < total and (now - self._last_log) < self.log_interval_s:
            return
        self._log(done)

    def _log(self, done: int) -> None:
        total = max(self.global_total, 1)
        elapsed = max(time.monotonic() - self._t0, 1e-6)
        rate = done / elapsed
        remain = (total - done) / rate if rate > 0 else 0.0
        self._last_log = time.monotonic()
        _LOG.info(
            "[%s] %d/%d  %.2f ex/s  elapsed %.0fs  eta %.0fs",
            self.desc,
            done,
            self.global_total,
            rate,
            elapsed,
            remain,
        )

    def close(self) -> None:
        if self.rank == 0 and self.global_total > 0:
            done = self.global_done()
            if done and (time.monotonic() - self._last_log) >= 0.2:
                self._log(done)
        if self.bar is not None:
            self.bar.n = min(self.global_done(), self.bar.total or 0)
            self.bar.refresh()
            self.bar.close()
            self.bar = None


@contextmanager
def overwrite_progress(
    total: int,
    desc: str,
    *,
    enabled: bool | None = None,
    mininterval: float = 0.3,
    global_total: int | None = None,
    progress_dir: str | Path | None = None,
    name: str = "eval",
    log_interval_s: float = 5.0,
) -> Iterator[ProgressHandle]:
    """Rank-0 tqdm on a real TTY; otherwise one INFO line every ``log_interval_s``."""
    from llm4rec.core import distributed as dist_utils

    local_total = max(0, int(total))
    world = max(1, dist_utils.world_size())
    rank = dist_utils.rank()
    gtotal = int(global_total) if global_total not in (None, 0) else local_total
    if enabled is None:
        enabled = dist_utils.is_main() and gtotal > 0 and sys.stderr.isatty()

    paths: list[Path] | None = None
    if progress_dir is not None and world > 1:
        root = Path(progress_dir)
        stem = str(name or "eval").replace("/", "_")
        paths = [root / f"{stem}.rank{i}" for i in range(world)]

    bar: tqdm | None = None
    if enabled and gtotal > 0:
        bar = tqdm(
            total=gtotal,
            desc=desc,
            file=sys.stderr,
            bar_format=_BAR_FORMAT,
            dynamic_ncols=True,
            mininterval=mininterval,
            leave=True,
        )
    handle = ProgressHandle(
        bar=bar,
        local_total=local_total,
        global_total=gtotal,
        rank=rank,
        world=world,
        paths=paths,
        desc=desc,
        log_interval_s=log_interval_s,
    )
    try:
        yield handle
    finally:
        handle.close()
