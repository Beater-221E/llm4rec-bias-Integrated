"""Poster downloader tests."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from llm4rec_bias_Integrated.data.mllm4rec.poster_downloader import (
    download_posters_from_matches,
    request_picture_original,
    request_picture_robust,
)


def _jpeg_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(buf, format="JPEG")
    return buf.getvalue()


def test_request_picture_original_writes(tmp_path: Path) -> None:
    dest = tmp_path / "1.jpg"
    data = _jpeg_bytes()

    class Resp:
        def getcode(self):
            return 200

        status = 200

        def read(self):
            return data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch(
        "llm4rec_bias_Integrated.data.mllm4rec.poster_downloader.urlopen",
        return_value=Resp(),
    ):
        assert request_picture_original("http://example/x.jpg", dest) is True
    assert dest.is_file()
    # resume: existing file skipped
    with patch("llm4rec_bias_Integrated.data.mllm4rec.poster_downloader.urlopen") as m:
        assert request_picture_original("http://example/x.jpg", dest) is True
        m.assert_not_called()


def test_robust_rejects_html(tmp_path: Path) -> None:
    dest = tmp_path / "2.jpg"

    class Resp:
        headers = {"Content-Type": "text/html"}

        def read(self):
            return b"<html>error</html>"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch(
        "llm4rec_bias_Integrated.data.mllm4rec.poster_downloader.urlopen",
        return_value=Resp(),
    ):
        ok, err = request_picture_robust(
            "http://example/x", dest, retries=1, verify_image=True
        )
    assert ok is False
    assert err and "html" in err


def test_download_from_matches_skips_empty_url(tmp_path: Path) -> None:
    matches = {
        1: {"poster_url": None, "match_status": "unmatched"},
        2: {"poster_url": "http://example/2.jpg", "match_status": "matched"},
    }
    data = _jpeg_bytes()

    class Resp:
        headers = {"Content-Type": "image/jpeg"}

        def getcode(self):
            return 200

        status = 200

        def read(self):
            return data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch(
        "llm4rec_bias_Integrated.data.mllm4rec.poster_downloader.urlopen",
        return_value=Resp(),
    ):
        stats = download_posters_from_matches(
            matches=matches,
            img_dir=tmp_path / "img",
            mode="original",
        )
    assert stats["skipped_no_url"] == 1
    assert stats["downloaded"] == 1
    assert (tmp_path / "img" / "2.jpg").is_file()
