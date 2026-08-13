"""SID 表：item ↔ 语义码，以及约束解码用的前缀树。

★ token 格式对齐**官方 MiniOneRec**：``<a_12><b_200><c_7>``

Collision-safe mapping:
  ``sid_to_items: dict[SID, list[ItemId]]`` — never silently overwrite.
  Unique-item decoding uses deterministic first-by-sorted-id resolution and logs
  when ambiguity occurs.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from llm4rec.core.exceptions import ConfigurationError, MissingArtifactError
from llm4rec.sid.artifact import SID_MAP_NAME, load_manifest

DEFAULT_PREFIXES = ("a", "b", "c", "d", "e")
_LOG = logging.getLogger(__name__)


def sid_token(layer: int, code: int, prefixes: Sequence[str] = DEFAULT_PREFIXES) -> str:
    if layer >= len(prefixes):
        raise ConfigurationError(f"SID 层数 {layer + 1} 超出前缀表 {list(prefixes)}")
    return f"<{prefixes[layer]}_{int(code)}>"


def format_sid(codes: Sequence[int], prefixes: Sequence[str] = DEFAULT_PREFIXES) -> str:
    return "".join(sid_token(i, c, prefixes) for i, c in enumerate(codes))


class SidTable:
    """从**静态产物目录**加载的 SID 表。只读。"""

    def __init__(self, sid_dir: str | Path) -> None:
        path = Path(sid_dir)
        map_path = path / SID_MAP_NAME
        if not map_path.is_file():
            raise MissingArtifactError(f"缺少 {map_path}")

        self.dir = path
        self.manifest = load_manifest(path)
        self.prefixes: tuple[str, ...] = tuple(
            self.manifest.layer_prefixes or DEFAULT_PREFIXES[: self.manifest.levels]
        )
        self.levels = int(self.manifest.levels)
        self.codebook_size = int(self.manifest.codebook_size)

        with map_path.open(encoding="utf-8") as fh:
            raw = json.load(fh)

        self.codes: dict[str, tuple[int, ...]] = {}
        for item_id, entry in raw.items():
            codes = entry["codes"] if isinstance(entry, dict) else entry
            self.codes[str(item_id)] = tuple(int(c) for c in codes)

        # Collision-safe reverse map: never overwrite earlier items.
        self.sid_to_items: dict[tuple[int, ...], list[str]] = {}
        for item, codes in self.codes.items():
            self.sid_to_items.setdefault(codes, []).append(item)
        for sid, items in self.sid_to_items.items():
            items.sort()  # deterministic order for decode

        n_unique = len(self.sid_to_items)
        if n_unique != len(self.codes):
            _LOG.warning(
                "SID collisions present: %d items → %d unique SIDs (%s)",
                len(self.codes),
                n_unique,
                path,
            )

        # Backward-compatible single-item view (deterministic first item).
        self.item_of: dict[tuple[int, ...], str] = {
            sid: items[0] for sid, items in self.sid_to_items.items()
        }

        self._pattern = re.compile(
            r"<(" + "|".join(re.escape(p) for p in self.prefixes) + r")_(\d+)>"
        )
        self._ambiguity_count = 0
        self._ambiguity_logged: set[tuple[int, ...]] = set()

    # ------------------------------------------------------------------ 编解码
    def sid(self, item_id: str | int) -> str:
        return format_sid(self.codes[str(item_id)], self.prefixes)

    def items_for_sid(self, codes: Sequence[int]) -> list[str]:
        return list(self.sid_to_items.get(tuple(int(c) for c in codes), []))

    def resolve_item(
        self, codes: Sequence[int], *, log_ambiguity: bool = False
    ) -> str | None:
        """Deterministic unique-item resolution for evaluation.

        When multiple catalog items share a SID, returns the lexicographically
        smallest item id. Ambiguity is counted on ``_ambiguity_count``; per-hit
        logging is off by default (eval spam).
        """
        items = self.items_for_sid(codes)
        if not items:
            return None
        if len(items) > 1:
            self._ambiguity_count += 1
            key = tuple(int(c) for c in codes)
            if log_ambiguity and key not in self._ambiguity_logged:
                self._ambiguity_logged.add(key)
                _LOG.debug(
                    "SID ambiguity: codes=%s maps to %d items; using id=%s",
                    key,
                    len(items),
                    items[0],
                )
        return items[0]

    def all_tokens(self) -> list[str]:
        """需要加进 tokenizer 的全部新 token。"""
        return [
            sid_token(layer, code, self.prefixes)
            for layer in range(self.levels)
            for code in range(self.codebook_size)
        ]

    def parse(self, text: str) -> str | None:
        """从生成文本里解析出 item id；解析不出返回 None。"""
        matches = self._pattern.findall(text)
        codes: list[int] = []
        for prefix, code in matches:
            expected = self.prefixes[len(codes)]
            if prefix != expected:
                if prefix == self.prefixes[0]:
                    codes = [int(code)]
                else:
                    codes = []
                continue
            codes.append(int(code))
            if len(codes) == self.levels:
                return self.resolve_item(codes)
        return None

    def parse_all(self, text: str) -> list[str]:
        """Return all items that share the parsed SID (empty if invalid)."""
        matches = self._pattern.findall(text)
        codes: list[int] = []
        for prefix, code in matches:
            expected = self.prefixes[len(codes)]
            if prefix != expected:
                if prefix == self.prefixes[0]:
                    codes = [int(code)]
                else:
                    codes = []
                continue
            codes.append(int(code))
            if len(codes) == self.levels:
                return self.items_for_sid(codes)
        return []

    # ------------------------------------------------------------- 约束解码
    def build_trie(self, tokenizer: Any, eos_id: int) -> dict:
        """全库 SID 的 token 前缀树，叶子挂 eos。"""
        root: dict = {}
        for codes in self.codes.values():
            ids = [
                tokenizer.convert_tokens_to_ids(sid_token(layer, code, self.prefixes))
                for layer, code in enumerate(codes)
            ]
            if any(i is None or i < 0 for i in ids):
                raise ConfigurationError(
                    "SID token 不在 tokenizer 词表里 —— "
                    "加载模型时必须先 add_tokens + resize_token_embeddings"
                )
            node = root
            for token_id in ids:
                node = node.setdefault(token_id, {})
            node[eos_id] = {}
        return root

    def prefix_allowed_fn(self, tokenizer: Any, prompt_len: int, eos_id: int):
        """给 ``model.generate(prefix_allowed_tokens_fn=...)`` 用。"""
        root = self.build_trie(tokenizer, eos_id)

        def fn(batch_id: int, input_ids: Any) -> list[int]:
            _ = batch_id
            node = root
            for token_id in input_ids[prompt_len:].tolist():
                node = node.get(token_id)
                if node is None:
                    return [eos_id]
            allowed = list(node.keys())
            return allowed or [eos_id]

        return fn

    # ------------------------------------------------------------------ 其它
    def collision_summary(self) -> dict[str, Any]:
        colliding = {sid: items for sid, items in self.sid_to_items.items() if len(items) > 1}
        return {
            "n_items": len(self.codes),
            "n_unique_sids": len(self.sid_to_items),
            "num_collision_groups": len(colliding),
            "max_collision_group_size": max((len(v) for v in colliding.values()), default=0),
            "ambiguity_resolutions_logged": self._ambiguity_count,
        }

    def items(self) -> Iterable[str]:
        return self.codes.keys()

    def __len__(self) -> int:
        return len(self.codes)

    def __contains__(self, item_id: object) -> bool:
        return str(item_id) in self.codes
