# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
#
# Original behavior is preserved unless explicitly documented.

"""Movie poster download (official ``requestPicture`` + robust extras)."""

from __future__ import annotations

import json
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("llm4rec_bias_Integrated.mllm4rec")

DownloadMode = Literal["original", "robust"]


def request_picture_original(image_url: str, save_image_path: Path) -> bool:
    """Official ``requestPicture``: skip if exists; GET once; write bytes; no retry."""
    save_image_path = Path(save_image_path)
    if save_image_path.exists():
        return True
    try:
        req = Request(image_url, headers={"Connection": "close"})
        with urlopen(req, timeout=60) as resp:
            if getattr(resp, "status", 200) != 200 and resp.getcode() != 200:
                return False
            data = resp.read()
        save_image_path.parent.mkdir(parents=True, exist_ok=True)
        with save_image_path.open("wb") as file_obj:
            file_obj.write(data)
        return True
    except Exception:  # noqa: BLE001 — official returns False
        return False


def request_picture_robust(
    image_url: str,
    save_image_path: Path,
    *,
    timeout_seconds: float = 30.0,
    retries: int = 3,
    overwrite: bool = False,
    verify_image: bool = True,
) -> tuple[bool, str | None]:
    """Retry + temp file + optional PIL verify; never raises."""
    save_image_path = Path(save_image_path)
    if save_image_path.exists() and not overwrite:
        if verify_image and not _pil_ok(save_image_path):
            save_image_path.unlink(missing_ok=True)
        else:
            return True, None

    last_err: str | None = None
    for attempt in range(max(1, retries)):
        tmp = save_image_path.with_suffix(save_image_path.suffix + ".tmp")
        try:
            req = Request(
                image_url,
                headers={"Connection": "close", "User-Agent": "llm4rec-bias-Integrated/mllm4rec"},
            )
            with urlopen(req, timeout=timeout_seconds) as resp:
                ctype = (resp.headers.get("Content-Type") or "").lower()
                data = resp.read()
            if "text/html" in ctype:
                last_err = f"html_content_type:{ctype}"
                time.sleep(0.5 * (2**attempt))
                continue
            if verify_image and not _pil_ok_bytes(data):
                last_err = "pil_verify_failed"
                time.sleep(0.5 * (2**attempt))
                continue
            save_image_path.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("wb") as fh:
                fh.write(data)
            tmp.replace(save_image_path)
            return True, None
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_err = str(exc)
            tmp.unlink(missing_ok=True)
            time.sleep(0.5 * (2**attempt))
    return False, last_err


def _pil_ok(path: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:  # noqa: BLE001
        return False


def _pil_ok_bytes(data: bytes) -> bool:
    try:
        from PIL import Image

        with Image.open(BytesIO(data)) as im:
            im.verify()
        return True
    except Exception:  # noqa: BLE001
        return False


def append_failure(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_posters_from_matches(
    *,
    matches: dict[int, dict[str, Any]],
    img_dir: Path,
    mode: DownloadMode = "original",
    timeout_seconds: float = 30.0,
    retries: int = 3,
    overwrite: bool = False,
    max_items: int | None = None,
    failed_log: Path | None = None,
) -> dict[str, int]:
    """Download ``img/{internal_id}.jpg`` from cached TMDb match records.

    Missing poster URL → skip (item kept). Failures logged; task continues.
    """
    img_dir = Path(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    failed_log = failed_log or (img_dir.parent / "failed_posters.jsonl")

    keys = list(matches.keys())
    if max_items is not None:
        keys = keys[: max(0, int(max_items))]

    from tqdm import tqdm

    stats = {"downloaded": 0, "skipped_exists": 0, "skipped_no_url": 0, "failed": 0}
    for iid in tqdm(keys, desc="Download posters", unit="item", dynamic_ncols=True):
        row = matches[iid]
        dest = img_dir / f"{iid}.jpg"
        url = row.get("poster_url") or ""
        if not url:
            stats["skipped_no_url"] += 1
            continue
        if dest.exists() and not overwrite:
            stats["skipped_exists"] += 1
            continue
        if mode == "original":
            ok = request_picture_original(url, dest)
            err = None if ok else "download_failed"
        else:
            ok, err = request_picture_robust(
                url,
                dest,
                timeout_seconds=timeout_seconds,
                retries=retries,
                overwrite=overwrite,
                verify_image=True,
            )
        if ok:
            stats["downloaded"] += 1
        else:
            stats["failed"] += 1
            append_failure(
                failed_log,
                {
                    "internal_item_id": iid,
                    "poster_url": url,
                    "error": err or "download_failed",
                },
            )
    logger.info("Poster download stats: %s", stats)
    return stats
