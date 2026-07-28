"""TMDb client unit tests (mocked HTTP)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm4rec_bias_Integrated.data.mllm4rec.tmdb_client import (
    TMDbAPIError,
    TMDbClient,
    match_dataset_meta,
    official_query_title,
    require_api_key,
)


def test_require_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    with pytest.raises(TMDbAPIError):
        require_api_key()


def test_official_query_strips_year() -> None:
    assert official_query_title("Toy Story (1995)") == "Toy Story"


def test_original_takes_first_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMDB_API_KEY", "test-key")
    client = TMDbClient(api_key="test-key", match_mode="original", retries=1)

    def fake_search(query: str):
        return [
            {"id": 1, "title": "Wrong", "release_date": "1990-01-01", "poster_path": "/a.jpg"},
            {"id": 2, "title": "Toy Story", "release_date": "1995-10-30", "poster_path": "/b.jpg"},
        ]

    def fake_details(movie_id: int):
        return {"id": movie_id, "title": "Wrong", "poster_path": "/a.jpg", "release_date": "1990-01-01"}

    client.search_movies = fake_search  # type: ignore[method-assign]
    client.movie_details = fake_details  # type: ignore[method-assign]
    rec = client.match_title(
        meta_title="Toy Story (1995)", internal_item_id=1, raw_item_id=10
    )
    assert rec.tmdb_id == 1
    assert rec.match_mode == "original"
    assert rec.poster_url and rec.poster_url.endswith("/a.jpg")


def test_robust_prefers_year(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMDB_API_KEY", "test-key")
    client = TMDbClient(api_key="test-key", match_mode="robust", retries=1)
    client.search_movies = lambda q: [  # type: ignore[method-assign]
        {"id": 1, "title": "Wrong", "release_date": "1990-01-01", "poster_path": "/a.jpg"},
        {"id": 862, "title": "Toy Story", "release_date": "1995-10-30", "poster_path": "/b.jpg"},
    ]
    client.movie_details = lambda mid: {  # type: ignore[method-assign]
        "id": mid,
        "title": "Toy Story",
        "poster_path": "/b.jpg",
        "release_date": "1995-10-30",
    }
    rec = client.match_title(
        meta_title="Toy Story (1995)", internal_item_id=1, raw_item_id=10
    )
    assert rec.tmdb_id == 862


def test_empty_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMDB_API_KEY", "test-key")
    client = TMDbClient(api_key="test-key", retries=1)
    client.search_movies = lambda q: []  # type: ignore[method-assign]
    rec = client.match_title(meta_title="Nope (1999)", internal_item_id=1, raw_item_id=1)
    assert rec.match_status == "unmatched"
    assert rec.poster_url is None


def test_find_img_original_empty_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMDB_API_KEY", "test-key")
    client = TMDbClient(api_key="test-key", retries=1)

    def boom(query: str):
        raise RuntimeError("network")

    client.search_movies = boom  # type: ignore[method-assign]
    assert client.find_poster_url_original("Toy Story (1995)") == ""


def test_match_cache_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMDB_API_KEY", "test-key")
    client = TMDbClient(api_key="test-key", retries=1)
    calls = {"n": 0}

    def search(q: str):
        calls["n"] += 1
        return [{"id": 9, "title": "X", "release_date": "1995-01-01", "poster_path": "/x.jpg"}]

    client.search_movies = search  # type: ignore[method-assign]
    client.movie_details = lambda mid: {  # type: ignore[method-assign]
        "id": mid,
        "title": "X",
        "poster_path": "/x.jpg",
        "release_date": "1995-01-01",
    }
    dataset = {
        "meta": {1: "A (1995)", 2: "B (1995)"},
        "smap": {100: 1, 200: 2},
    }
    cache = tmp_path / "tmdb_matches.jsonl"
    match_dataset_meta(dataset, client=client, cache_path=cache, resume=True)
    assert calls["n"] == 2
    match_dataset_meta(dataset, client=client, cache_path=cache, resume=True)
    assert calls["n"] == 2  # resumed
    lines = cache.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
