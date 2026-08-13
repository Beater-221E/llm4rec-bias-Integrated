"""Optional sequence-length bucketing for integrated SFT (performance mode).

Status: **experimental_inactive**. The sampler exists but is not wired into
``run_sft`` / DPO. Prefer HF Trainer ``group_by_length`` for SFT when desired.
Do not activate purely for MiniOneRec reproduction.
"""

from __future__ import annotations

from typing import Any, Sequence

from torch.utils.data import Sampler


DEFAULT_BUCKETS = (128, 256, 384, 512, 1024)
BUCKETING_STATUS = "experimental_inactive"


def choose_bucket(length: int, buckets: Sequence[int]) -> int:
    for b in buckets:
        if length <= b:
            return int(b)
    return int(buckets[-1])


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Group similar-length indices into batches (integrated/opt-in only)."""

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        batch_size: int,
        buckets: Sequence[int] = DEFAULT_BUCKETS,
        drop_last: bool = False,
    ) -> None:
        self.lengths = list(lengths)
        self.batch_size = max(1, int(batch_size))
        self.buckets = tuple(int(b) for b in buckets)
        self.drop_last = drop_last
        self._batches = self._build()

    def _build(self) -> list[list[int]]:
        groups: dict[int, list[int]] = {b: [] for b in self.buckets}
        for idx, length in enumerate(self.lengths):
            b = choose_bucket(int(length), self.buckets)
            groups[b].append(idx)
        batches: list[list[int]] = []
        for b in self.buckets:
            ids = groups[b]
            for start in range(0, len(ids), self.batch_size):
                chunk = ids[start : start + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                if chunk:
                    batches.append(chunk)
        return batches

    def __iter__(self):
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


def enabled_bucketing(cfg: dict[str, Any]) -> bool:
    opt = (cfg.get("optimization") or {}).get("bucketing") or {}
    if not opt:
        return False
    if str(cfg.get("mode") or "").lower() == "reproduction":
        return False
    return bool(opt.get("enabled", False))
