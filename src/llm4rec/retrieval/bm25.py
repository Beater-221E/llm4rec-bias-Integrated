"""BM25 检索器 —— Rec-R1 路线用。

原文用 Pyserini/Lucene 多字段索引。这里是纯 Python 实现：Amazon 单类目
~1-2 万 item 的量级完全够用，省掉 Java + pyserini 依赖。

接口是 ``Retriever`` 协议，要换成真 Lucene 只需另写一个实现，
decoder 和 reward 都不用动。

★ 查询语法：原文的 query 支持 Lucene 布尔语法（``NOT "xxx" AND yyy``）。
  纯 Python 版做了简化处理：识别 ``NOT "短语"`` 作为负向过滤，
  其余的 AND/OR 当普通词项。完整 Lucene 语义请切 ``kind: lucene``。
"""

from __future__ import annotations

import json
import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from llm4rec.core.exceptions import MissingArtifactError

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NOT_RE = re.compile(r'\bNOT\s+"([^"]+)"|\bNOT\s+(\S+)')
_OPERATOR_RE = re.compile(r"\b(AND|OR|NOT)\b")


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int) -> list[str]: ...


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


def parse_query(query: str) -> tuple[list[str], list[str]]:
    """拆出 ``(正向词, 负向词)``。"""
    negatives: list[str] = []
    for quoted, bare in _NOT_RE.findall(query):
        negatives.extend(tokenize(quoted or bare))
    positive_text = _NOT_RE.sub(" ", query)
    positive_text = _OPERATOR_RE.sub(" ", positive_text)
    return tokenize(positive_text), negatives


@dataclass
class BM25Index:
    """经典 BM25（Okapi）。多字段用加权拼接实现。"""

    item_ids: list[str]
    doc_freq: dict[str, int]
    term_docs: dict[str, list[tuple[int, int]]]  # term → [(doc_idx, tf)]
    doc_len: list[int]
    avg_len: float
    k1: float = 0.9
    b: float = 0.4

    def search(self, query: str, *, top_k: int = 100) -> list[str]:
        positives, negatives = parse_query(query)
        if not positives:
            return []

        n_docs = len(self.item_ids)
        scores: dict[int, float] = defaultdict(float)
        for term in positives:
            postings = self.term_docs.get(term)
            if not postings:
                continue
            df = self.doc_freq[term]
            # Okapi BM25 的 idf，加 0.5 平滑避免 df 接近 N 时为负
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for doc_idx, tf in postings:
                norm = 1.0 - self.b + self.b * self.doc_len[doc_idx] / self.avg_len
                scores[doc_idx] += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)

        if negatives:
            banned: set[int] = set()
            for term in negatives:
                for doc_idx, _ in self.term_docs.get(term, ()):
                    banned.add(doc_idx)
            for doc_idx in banned:
                scores.pop(doc_idx, None)

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [self.item_ids[idx] for idx, _ in ranked]

    # ------------------------------------------------------------ 持久化
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        if not Path(path).is_file():
            raise MissingArtifactError(
                f"缺少 BM25 索引 {path}，先跑 `STEPS=bm25 bash prepare.sh`"
            )
        with Path(path).open("rb") as fh:
            return pickle.load(fh)


def build_index(
    item_meta: dict[str, dict[str, Any]],
    *,
    fields: Sequence[str] = ("title", "description", "categories", "brand"),
    field_weights: dict[str, int] | None = None,
    k1: float = 0.9,
    b: float = 0.4,
    log: Any = print,
) -> BM25Index:
    """建索引。多字段通过"重复词项"实现加权（title 比 description 重要）。"""
    weights = field_weights or {"title": 3, "brand": 2, "categories": 2, "description": 1}
    item_ids = sorted(item_meta.keys())

    doc_freq: Counter[str] = Counter()
    term_docs: dict[str, list[tuple[int, int]]] = defaultdict(list)
    doc_len: list[int] = []

    for idx, item in enumerate(item_ids):
        meta = item_meta[item]
        tokens: list[str] = []
        for field in fields:
            value = meta.get(field)
            if not value:
                continue
            tokens.extend(tokenize(value) * int(weights.get(field, 1)))
        counts = Counter(tokens)
        doc_len.append(len(tokens))
        for term, tf in counts.items():
            term_docs[term].append((idx, tf))
            doc_freq[term] += 1
        if idx % 5000 == 0 and idx:
            log(f"[bm25]   {idx}/{len(item_ids)}")

    avg_len = sum(doc_len) / max(len(doc_len), 1)
    log(f"[bm25] 索引完成：{len(item_ids)} 物品，{len(term_docs)} 词项，平均长度 {avg_len:.1f}")
    return BM25Index(
        item_ids=item_ids,
        doc_freq=dict(doc_freq),
        term_docs=dict(term_docs),
        doc_len=doc_len,
        avg_len=avg_len or 1.0,
        k1=k1,
        b=b,
    )


def index_path_for(cfg: dict[str, Any]) -> Path:
    from llm4rec.data.base import get_adapter

    retr = (cfg.get("decoder") or {}).get("retriever") or {}
    root = Path(retr.get("index_dir") or "artifacts/bm25")
    return root / get_adapter(cfg).dataset_key(cfg) / "bm25.pkl"


def build_and_save(cfg: dict[str, Any], *, force: bool = False, log: Any = print) -> Path:
    from llm4rec.data.base import get_adapter

    path = index_path_for(cfg)
    if path.is_file() and not force:
        log(f"[bm25] 已存在，跳过：{path}")
        return path

    retr = (cfg.get("decoder") or {}).get("retriever") or {}
    meta = get_adapter(cfg).load_item_meta(cfg)
    index = build_index(
        meta,
        fields=tuple(retr.get("fields") or ("title", "description", "categories", "brand")),
        k1=float(retr.get("k1") or 0.9),
        b=float(retr.get("b") or 0.4),
        log=log,
    )
    index.save(path)
    (path.parent / "stats.json").write_text(
        json.dumps(
            {"n_items": len(index.item_ids), "n_terms": len(index.term_docs), "avg_len": index.avg_len},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"[bm25] 完成 → {path}")
    return path
