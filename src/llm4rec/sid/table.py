"""SID 表：item ↔ 语义码，以及约束解码用的前缀树。

★ token 格式对齐**官方 MiniOneRec**：``<a_12><b_200><c_7>``
  （官方 ``rq/generate_indices.py`` 的 ``prefix = ["<a_{}>","<b_{}>",...]``，
   mor-reproduce 的 ``sid/codec.py`` 也是这个格式）

原 Integrated 仓库用的是 ``<s0_12><s1_200>``，和官方/mor-reproduce 都对不上。
换成官方格式之后，mor-reproduce 已经训好的 checkpoint 的新增 embedding
才能原样对上词表，不用重训。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from llm4rec.core.exceptions import ConfigurationError, MissingArtifactError
from llm4rec.sid.artifact import SID_MAP_NAME, load_manifest

DEFAULT_PREFIXES = ("a", "b", "c", "d", "e")


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

        self.item_of: dict[tuple[int, ...], str] = {
            codes: item for item, codes in self.codes.items()
        }
        if len(self.item_of) != len(self.codes):
            raise ConfigurationError(
                f"{path} 里的 SID 存在碰撞（{len(self.codes)} 个物品只映射到 "
                f"{len(self.item_of)} 个唯一 SID）。产物不该出现这种情况，请重建。"
            )

        self._pattern = re.compile(
            r"<(" + "|".join(re.escape(p) for p in self.prefixes) + r")_(\d+)>"
        )

    # ------------------------------------------------------------------ 编解码
    def sid(self, item_id: str | int) -> str:
        return format_sid(self.codes[str(item_id)], self.prefixes)

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
                # 层序错乱 → 这一段不是合法 SID，重头再找
                if prefix == self.prefixes[0]:
                    codes = [int(code)]
                else:
                    codes = []
                continue
            codes.append(int(code))
            if len(codes) == self.levels:
                return self.item_of.get(tuple(codes))
        return None

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
        """给 ``model.generate(prefix_allowed_tokens_fn=...)`` 用。

        保证每条 beam 都走在前缀树上 —— 也就是官方说的
        "every beam is unique and valid"，非法 SID 率恒为 0。
        """
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
    def items(self) -> Iterable[str]:
        return self.codes.keys()

    def __len__(self) -> int:
        return len(self.codes)

    def __contains__(self, item_id: object) -> bool:
        return str(item_id) in self.codes
