"""Semantic ID table, tokens, parsing, and constrained-decoding trie."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SID_RE = re.compile(r"<s(\d+)_(\d+)>")


def sid_string(codes: list[int] | tuple[int, ...]) -> str:
    return "".join(f"<s{level}_{code}>" for level, code in enumerate(codes))


def all_sid_tokens(levels: int, K: int, collision_K: int) -> list[str]:
    """levels = residual levels (collision level is levels)."""
    toks = [f"<s{level}_{c}>" for level in range(levels) for c in range(K)]
    toks += [f"<s{levels}_{c}>" for c in range(collision_K)]
    return toks


def parse_sid(text: str, num_levels: int) -> tuple[int, ...] | None:
    """First ``num_levels`` well-ordered ``<sL_C>`` tokens → codes tuple."""
    hits = SID_RE.findall(text)
    codes: list[int] = []
    for level, code in hits:
        if int(level) == len(codes):
            codes.append(int(code))
            if len(codes) == num_levels:
                return tuple(codes)
        else:
            break
    return None


class SidTable:
    """item_id (str) ↔ semantic ID codes + trie helpers."""

    def __init__(self, path: str | Path) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.path = Path(path)
        self.levels = int(data["levels"]) + 1  # + collision
        self.K = int(data["K"])
        self.collision_K = int(data["collision_K"])
        self.encoder = str(data.get("encoder") or "")
        self.codes: dict[str, tuple[int, ...]] = {
            str(i): tuple(int(x) for x in c) for i, c in data["items"].items()
        }
        self.item_of: dict[tuple[int, ...], str] = {c: i for i, c in self.codes.items()}

    def sid(self, item_id: str | int) -> str:
        return sid_string(self.codes[str(item_id)])

    def tokens(self) -> list[str]:
        return all_sid_tokens(self.levels - 1, self.K, self.collision_K)

    def parse(self, text: str) -> str | None:
        codes = parse_sid(text, self.levels)
        if codes is None:
            return None
        return self.item_of.get(codes)

    def trie(self, tokenizer: Any, eos_id: int) -> dict:
        root: dict = {}
        for codes in self.codes.values():
            ids = [
                tokenizer.convert_tokens_to_ids(f"<s{level}_{c}>")
                for level, c in enumerate(codes)
            ]
            node = root
            for t in ids:
                node = node.setdefault(t, {})
            node[eos_id] = {}
        return root

    def prefix_fn(self, tokenizer: Any, prompt_len: int, eos_id: int):
        root = self.trie(tokenizer, eos_id)

        def fn(batch_id, input_ids):  # noqa: ANN001
            _ = batch_id
            node = root
            for t in input_ids[prompt_len:].tolist():
                node = node.get(t)
                if node is None:
                    return [eos_id]
            return list(node.keys()) or [eos_id]

        return fn

    def to_dict(self) -> dict[str, Any]:
        return {
            "levels": self.levels - 1,
            "K": self.K,
            "collision_K": self.collision_K,
            "encoder": self.encoder,
            "items": {i: list(c) for i, c in self.codes.items()},
        }
