"""Generation helpers (placeholder for Phase 5+)."""

from __future__ import annotations

from typing import Any


def default_generation_kwargs(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or {}
    return {
        "max_new_tokens": int(cfg.get("max_new_tokens", cfg.get("max_completion_length", 16))),
        "do_sample": bool(cfg.get("do_sample", False)),
        "temperature": float(cfg.get("temperature", 1.0)),
        "top_p": float(cfg.get("top_p", 1.0)),
    }
