"""Qwen2.5 model alias table."""

from __future__ import annotations

QWEN_ALIASES: dict[str, str] = {
    # canonical instruct checkpoints
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen25_0_5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen25_0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-1b": "Qwen/Qwen2.5-1.5B-Instruct",  # compat alias
    "qwen25_1_5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen25_3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen25_7b": "Qwen/Qwen2.5-7B-Instruct",
}


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def resolve_qwen_checkpoint(name: str) -> str | None:
    if not name:
        return None
    key = _norm(name)
    if key in QWEN_ALIASES:
        return QWEN_ALIASES[key]
    # also try underscore form keys
    alt = name.strip().lower().replace("-", "_")
    for k, v in QWEN_ALIASES.items():
        if k.replace("-", "_") == alt:
            return v
    if name.startswith("Qwen/") or name.startswith("qwen/"):
        return name
    return None
