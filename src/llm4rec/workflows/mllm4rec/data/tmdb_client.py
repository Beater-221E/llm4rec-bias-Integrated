# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
#
# Original behavior is preserved unless explicitly documented.
# TMDb API key MUST come from the environment (never hardcode).

"""TMDb movie search / poster URL resolution."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from llm4rec.workflows.mllm4rec.data.constants import TMDB_API_KEY_ENV, TMDB_IMAGE_BASE_URL

logger = logging.getLogger("llm4rec.workflows.mllm4rec._stack")

MatchMode = Literal["original", "robust"]


@dataclass
class TMDbMatchRecord:
    raw_item_id: Any
    internal_item_id: int
    raw_title: str
    query_title: str
    release_year: int | None
    tmdb_id: int | None
    tmdb_title: str | None
    tmdb_release_date: str | None
    poster_path: str | None
    poster_url: str | None
    match_status: str
    match_mode: str
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class TMDbAPIError(RuntimeError):
    pass


def require_api_key(env_name: str = TMDB_API_KEY_ENV) -> str:
    key = os.environ.get(env_name)
    if key is not None:
        key = key.strip()
    if not key:
        raise TMDbAPIError(
            f"缺少 TMDb API 密钥：环境变量 {env_name} 未设置或为空。\n"
            f"请先执行：\n"
            f'  export {env_name}="你的TMDb_v3_API_Key"\n'
            f"或写入 ~/.bashrc 后执行 source ~/.bashrc。\n"
            f"申请地址：https://www.themoviedb.org/settings/api\n"
            f"注意：不要把密钥写进 YAML / 代码 / git。"
        )
    return key


def official_query_title(meta_title: str) -> str:
    """Official ``find_img`` query: ``movie_name[:-7]`` (strip year suffix)."""
    if len(meta_title) < 7:
        return meta_title
    return meta_title[:-7]


def parse_year_from_meta_title(meta_title: str) -> int | None:
    m = re.search(r"\((\d{4})\)\s*$", meta_title)
    return int(m.group(1)) if m else None


class TMDbClient:
    """Thin TMDb REST client (no hardcoded secrets)."""

    SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
    MOVIE_URL = "https://api.themoviedb.org/3/movie/{movie_id}"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = TMDB_API_KEY_ENV,
        image_base_url: str = TMDB_IMAGE_BASE_URL,
        timeout_seconds: float = 30.0,
        retries: int = 3,
        match_mode: MatchMode = "original",
    ) -> None:
        self.api_key = api_key or require_api_key(api_key_env)
        self.image_base_url = image_base_url.rstrip("/") + "/"
        self.timeout_seconds = float(timeout_seconds)
        self.retries = max(1, int(retries))
        self.match_mode = match_mode

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        qs = urlencode({**params, "api_key": self.api_key})
        full = f"{url}?{qs}"
        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                # Prefer requests when available (more resilient TLS than bare urlopen).
                try:
                    import requests

                    resp = requests.get(
                        url,
                        params={**params, "api_key": self.api_key},
                        timeout=self.timeout_seconds,
                        headers={"User-Agent": "llm4rec-bias-Integrated/mllm4rec"},
                    )
                    resp.raise_for_status()
                    return resp.json()
                except ImportError:
                    req = Request(full, headers={"User-Agent": "llm4rec-bias-Integrated/mllm4rec"})
                    with urlopen(req, timeout=self.timeout_seconds) as resp:
                        return json.loads(resp.read().decode("utf-8"))
            except Exception as exc:  # noqa: BLE001 — retried below
                last_err = exc
                sleep_s = 0.5 * (2**attempt)
                logger.warning(
                    "TMDb request failed (attempt %s/%s): %s; sleep %.1fs",
                    attempt + 1,
                    self.retries,
                    exc,
                    sleep_s,
                )
                time.sleep(sleep_s)
        raise TMDbAPIError(f"TMDb request failed after retries: {last_err}")

    def search_movies(self, query: str) -> list[dict[str, Any]]:
        data = self._get_json(self.SEARCH_URL, {"query": query, "include_adult": "false"})
        return list(data.get("results") or [])

    def movie_details(self, movie_id: int) -> dict[str, Any]:
        return self._get_json(
            self.MOVIE_URL.format(movie_id=movie_id),
            {"append_to_response": "images"},
        )

    def poster_url_from_path(self, poster_path: str | None) -> str | None:
        if not poster_path:
            return None
        path = poster_path if str(poster_path).startswith("/") else f"/{poster_path}"
        return self.image_base_url.rstrip("/") + path

    def find_poster_url_original(self, meta_title: str) -> str:
        """Official ``find_img``: first search hit → details → poster URL, else ``\"\"``."""
        try:
            query = official_query_title(meta_title)
            movies = self.search_movies(query)
            movie_id = int(movies[0]["id"])
            movie = self.movie_details(movie_id)
            poster_path = movie.get("poster_path")
            return self.poster_url_from_path(poster_path) or ""
        except Exception as exc:  # noqa: BLE001 — official bare except → ""
            logger.debug("original find_img failed for %r: %s", meta_title, exc)
            return ""

    def match_title(
        self,
        *,
        meta_title: str,
        internal_item_id: int,
        raw_item_id: Any,
    ) -> TMDbMatchRecord:
        year = parse_year_from_meta_title(meta_title)
        query = official_query_title(meta_title)
        mode = self.match_mode
        try:
            results = self.search_movies(query)
        except TMDbAPIError as exc:
            return TMDbMatchRecord(
                raw_item_id=raw_item_id,
                internal_item_id=internal_item_id,
                raw_title=meta_title,
                query_title=query,
                release_year=year,
                tmdb_id=None,
                tmdb_title=None,
                tmdb_release_date=None,
                poster_path=None,
                poster_url=None,
                match_status="error",
                match_mode=mode,
                error=str(exc),
            )

        if not results:
            return TMDbMatchRecord(
                raw_item_id=raw_item_id,
                internal_item_id=internal_item_id,
                raw_title=meta_title,
                query_title=query,
                release_year=year,
                tmdb_id=None,
                tmdb_title=None,
                tmdb_release_date=None,
                poster_path=None,
                poster_url=None,
                match_status="unmatched",
                match_mode=mode,
                error="empty_search",
            )

        # original: always first hit (official). robust: year-aware pick when possible.
        chosen = results[0]
        if mode == "robust" and year is not None:
            chosen = _robust_pick(results, year) or chosen

        tmdb_id = int(chosen["id"])
        # Engineering: if search already has poster_path, skip details call.
        # Movie choice stays identical (first hit / robust pick). Official find_img
        # also hits details; we keep that path in find_poster_url_original().
        details = chosen
        if not chosen.get("poster_path"):
            try:
                details = self.movie_details(tmdb_id)
            except TMDbAPIError:
                details = chosen
        poster_path = details.get("poster_path") or chosen.get("poster_path")
        poster_url = self.poster_url_from_path(poster_path)
        status = "matched" if poster_url else "matched_no_poster"
        return TMDbMatchRecord(
            raw_item_id=raw_item_id,
            internal_item_id=internal_item_id,
            raw_title=meta_title,
            query_title=query,
            release_year=year,
            tmdb_id=tmdb_id,
            tmdb_title=details.get("title") or chosen.get("title"),
            tmdb_release_date=details.get("release_date") or chosen.get("release_date"),
            poster_path=poster_path,
            poster_url=poster_url,
            match_status=status,
            match_mode=mode,
            error=None,
        )


def _robust_pick(results: list[dict[str, Any]], year: int) -> dict[str, Any] | None:
    exact: list[dict[str, Any]] = []
    near: list[dict[str, Any]] = []
    for r in results:
        rd = (r.get("release_date") or "")[:4]
        if not rd.isdigit():
            continue
        y = int(rd)
        if y == year:
            exact.append(r)
        elif abs(y - year) <= 1:
            near.append(r)
    if exact:
        return exact[0]
    if near:
        return near[0]
    return None


def load_match_cache(path: Path) -> dict[int, dict[str, Any]]:
    cache: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return cache
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cache[int(row["internal_item_id"])] = row
    return cache


def append_match_record(path: Path, record: TMDbMatchRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")


def rewrite_match_cache(path: Path, records: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for iid in sorted(records):
            fh.write(json.dumps(records[iid], ensure_ascii=False) + "\n")
    tmp.replace(path)


def match_dataset_meta(
    dataset: dict[str, Any],
    *,
    client: TMDbClient,
    cache_path: Path,
    resume: bool = True,
    overwrite: bool = False,
    max_items: int | None = None,
) -> dict[int, dict[str, Any]]:
    """Match ``meta`` items into ``tmdb_matches.jsonl`` (resume-safe)."""
    from tqdm import tqdm

    cache = load_match_cache(cache_path) if (resume and not overwrite) else {}
    if overwrite and cache_path.is_file():
        cache_path.unlink()
        cache = {}

    inv_smap = {int(v): k for k, v in (dataset.get("smap") or {}).items()}
    meta: dict[int, str] = dataset["meta"]
    items = list(meta.items())
    if max_items is not None:
        items = items[: max(0, int(max_items))]

    todo: list[tuple[int, str]] = []
    for key, title in items:
        iid = int(key)
        if resume and iid in cache and not overwrite:
            continue
        todo.append((iid, title))

    skipped = len(items) - len(todo)
    logger.info(
        "TMDb match: total=%s cached/skip=%s remaining=%s",
        len(items),
        skipped,
        len(todo),
    )

    for iid, title in tqdm(todo, desc="TMDb match", unit="item", dynamic_ncols=True):
        rec = client.match_title(
            meta_title=title,
            internal_item_id=iid,
            raw_item_id=inv_smap.get(iid),
        )
        append_match_record(cache_path, rec)
        cache[iid] = rec.to_json()
        # Light pacing to reduce TLS / rate-limit failures on long runs.
        time.sleep(0.05)

    logger.info("TMDb match cache size=%s path=%s", len(cache), cache_path)
    return cache
