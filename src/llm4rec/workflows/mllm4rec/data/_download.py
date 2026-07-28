# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
#
# Original behavior is preserved unless explicitly documented.

"""Download / unzip helpers used by MovieLens dataset classes."""

from __future__ import annotations

import logging
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger("llm4rec.workflows.mllm4rec._stack")


def download(url: str, savepath: Path) -> None:
    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, savepath)
    urlretrieve(url, str(savepath))


def unzip(zippath: Path, savepath: Path) -> None:
    logger.info("Extracting %s -> %s", zippath, savepath)
    savepath = Path(savepath)
    savepath.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zippath) as zf:
        zf.extractall(savepath)


def download_and_extract_zip_folder(
    url: str,
    dest_folder: Path,
    *,
    zip_content_is_folder: bool = True,
) -> Path:
    """Download a zip whose top-level entry is a folder, move it to ``dest_folder``.

    Mirrors official ``maybe_download_raw_dataset`` for ml-latest-small.
    """
    import os
    import shutil

    dest_folder = Path(dest_folder)
    tmproot = Path(tempfile.mkdtemp())
    try:
        tmpzip = tmproot / "file.zip"
        tmpfolder = tmproot / "folder"
        download(url, tmpzip)
        unzip(tmpzip, tmpfolder)
        if zip_content_is_folder:
            children = list(tmpfolder.iterdir())
            if len(children) != 1:
                # Still take first entry like os.listdir(...)[0] in official code.
                inner = tmpfolder / os.listdir(tmpfolder)[0]
            else:
                inner = children[0]
            tmpfolder = inner
        if dest_folder.exists():
            shutil.rmtree(dest_folder)
        shutil.move(str(tmpfolder), str(dest_folder))
    finally:
        shutil.rmtree(tmproot, ignore_errors=True)
    return dest_folder
