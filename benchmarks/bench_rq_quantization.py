#!/usr/bin/env python
"""Benchmark RQ nearest-code quantization (reference backend).

Usage:
  PYTHONPATH=src python benchmarks/bench_rq_quantization.py
"""

from __future__ import annotations

import json
import time

import torch

from llm4rec.kernels import quantize_nearest


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    codebook_sizes = [256, 512]
    dims = [32, 64]
    rows = [256, 2048, 20480, 100_000]
    results: list[dict[str, float]] = []
    for k in codebook_sizes:
        for d in dims:
            cb = torch.randn(k, d, device=device)
            for n in rows:
                x = torch.randn(n, d, device=device)
                if device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                reps = 20
                for _ in range(reps):
                    quantize_nearest(x, cb, backend="reference")
                if device == "cuda":
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / reps
                peak_mib = (
                    torch.cuda.max_memory_allocated() / (1024**2)
                    if device == "cuda"
                    else None
                )
                rec = {
                    "device": device,
                    "n": n,
                    "codebook": k,
                    "dim": d,
                    "mean_s": round(dt, 6),
                    "items_per_s": round(n / dt, 1),
                }
                if peak_mib is not None:
                    rec["peak_mib"] = round(peak_mib, 1)
                results.append(rec)
                print(json.dumps(rec))
    print(
        json.dumps(
            {
                "note": "Compare these times with total SID preprocess (RQ-VAE train). "
                "Implement Triton only if this op is a measured hotspot."
            }
        )
    )


if __name__ == "__main__":
    main()
