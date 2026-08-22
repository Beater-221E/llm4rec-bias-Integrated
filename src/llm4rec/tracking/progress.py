"""Rank-0 progress: one tqdm line that overwrites itself.

``run.sh | tee`` makes stdout/stderr pipes, so a bar on stderr would emit a
new line every refresh. The bar is therefore drawn on ``/dev/tty`` (the
controlling terminal) and never logged. No TTY at all (nohup) falls back to
a throttled INFO line.

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
from typing import Iterator, TextIO

from tqdm import tqdm

_LOG = logging.getLogger(__name__)
_BAR_FORMAT = "{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def _open_overwrite_stream() -> tuple[TextIO | None, bool]:
    """Prefer the controlling TTY so ``\\r`` overwrites even under ``tee``."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        if sys.stderr.isatty():
            return sys.stderr, False
        return None, False
    if sys.stderr.isatty():
        return sys.stderr, False
    try:
        stream = open("/dev/tty", "w", encoding="utf-8", buffering=1)
    except OSError:
        return None, False
    if stream.isatty():
        return stream, True
    stream.close()
    return None, False


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
        if self.bar is None and self.rank == 0 and self.global_total > 0:
            self._log(0)

    def update(self, n: int = 1) -> int:
        self.local_done += int(n)
        if self.paths:
            _atomic_write_int(self.paths[self.rank], self.local_done)
        done = self.global_done()
        if self.bar is not None:
            self.bar.n = min(done, self.bar.total or done)
            self.bar.refresh()
            return done
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
        if self.bar is None and self.rank == 0 and self.global_total > 0:
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
    file: TextIO | None = None,
) -> Iterator[ProgressHandle]:
    """Rank-0 tqdm on the controlling TTY; otherwise a throttled INFO fallback."""
    from llm4rec.core import distributed as dist_utils

    local_total = max(0, int(total))
    world = max(1, dist_utils.world_size())
    rank = dist_utils.rank()
    gtotal = int(global_total) if global_total not in (None, 0) else local_total
    owned_stream = False
    stream: TextIO | None = file
    if enabled is False or not dist_utils.is_main() or gtotal <= 0:
        stream = None
    elif stream is None:
        stream, owned_stream = _open_overwrite_stream()
        if enabled is True and stream is None:
            stream = sys.stderr
            owned_stream = False
    draw_bar = stream is not None

    paths: list[Path] | None = None
    if progress_dir is not None and world > 1:
        root = Path(progress_dir)
        stem = str(name or "eval").replace("/", "_")
        paths = [root / f"{stem}.rank{i}" for i in range(world)]

    bar: tqdm | None = None
    if draw_bar:
        bar = tqdm(
            total=gtotal,
            desc=desc,
            file=stream,
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
        if owned_stream and stream is not None:
            stream.close()
