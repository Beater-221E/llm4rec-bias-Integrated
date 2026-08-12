"""Optional wall-clock / torch.profiler helpers for throughput metrics."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class PhaseTimer:
    enabled: bool = False
    cuda_sync: bool = False
    use_cuda_events: bool = False
    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        torch = _torch()
        use_events = (
            self.use_cuda_events
            and torch is not None
            and torch.cuda.is_available()
        )
        if use_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                end.synchronize()
                dt = float(start.elapsed_time(end)) / 1000.0
                self.totals[name] = self.totals.get(name, 0.0) + dt
                self.counts[name] = self.counts.get(name, 0) + 1
            return
        if self.cuda_sync and torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.cuda_sync and torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, total in self.totals.items():
            n = max(1, self.counts.get(k, 1))
            out[k] = {"total_s": round(total, 4), "count": n, "mean_s": round(total / n, 4)}
        return out


def _torch():
    try:
        import torch

        return torch
    except Exception:  # noqa: BLE001
        return None


def peak_vram_gb() -> float | None:
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / (1024**3), 3)


def make_timer(cfg: dict[str, Any]) -> PhaseTimer:
    prof = cfg.get("profiling") or {}
    enabled = bool(prof.get("enabled", False))
    # Prefer CUDA Events for GPU-heavy phases when profiling is on.
    use_events = bool(prof.get("cuda_events", enabled))
    cuda_sync = bool(prof.get("cuda_sync", False)) and not use_events
    return PhaseTimer(enabled=enabled, cuda_sync=cuda_sync, use_cuda_events=use_events)


def make_scheduled_profiler(cfg: dict[str, Any], *, output_dir: str, rank: int = 0):
    """Return a scheduled torch.profiler or None. Only rank 0 exports traces."""
    prof_cfg = cfg.get("profiling") or {}
    if not prof_cfg.get("enabled"):
        return None
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        from torch.profiler import (
            ProfilerActivity,
            profile,
            schedule,
            tensorboard_trace_handler,
        )
    except Exception:  # noqa: BLE001
        return None
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sched = schedule(
        wait=int(prof_cfg.get("wait_steps") or 1),
        warmup=int(prof_cfg.get("warmup_steps") or 2),
        active=int(prof_cfg.get("active_steps") or 5),
        repeat=int(prof_cfg.get("repeat") or 1),
    )
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    handler = tensorboard_trace_handler(str(out)) if rank == 0 else None
    kwargs = {"activities": activities, "schedule": sched, "record_shapes": False}
    if handler is not None:
        kwargs["on_trace_ready"] = handler
    try:
        return profile(**kwargs)
    except Exception:  # noqa: BLE001
        kwargs.pop("on_trace_ready", None)
        try:
            return profile(**kwargs)
        except Exception:  # noqa: BLE001
            return None


@contextmanager
def optional_torch_profiler(cfg: dict[str, Any], output_dir: str | None = None):
    """Yield a torch.profiler when profiling.enabled and CUDA available."""
    prof_cfg = cfg.get("profiling") or {}
    if not prof_cfg.get("enabled"):
        yield None
        return
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        yield None
        return
    try:
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(activities=activities, record_shapes=False) as prof:
            yield prof
        if output_dir:
            from pathlib import Path

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(Path(output_dir) / "chrome_trace.json"))
    except Exception:  # noqa: BLE001
        yield None
