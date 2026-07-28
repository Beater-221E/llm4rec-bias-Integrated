"""Terminal progress bar via a dedicated TTY (not the redirected log streams).

Runner scripts export ``LLM4REC_PROGRESS_TTY`` to a concrete ``/dev/pts/N`` path
(and optionally ``LLM4REC_PROGRESS_FD=3``). Accelerate elastic workers often have
no controlling terminal, so opening bare ``/dev/tty`` fails — the pts path works.
"""

from __future__ import annotations

import os
import sys
from typing import Any, TextIO


_PATCHED = False
_TTY_STREAM: TextIO | None = None


def get_progress_stream() -> TextIO | None:
    """Return the live terminal stream for progress, or ``None``."""
    global _TTY_STREAM
    if _TTY_STREAM is not None:
        try:
            if not _TTY_STREAM.closed:
                return _TTY_STREAM
        except Exception:
            _TTY_STREAM = None
    _TTY_STREAM = _open_progress_tty()
    return _TTY_STREAM


def write_progress_status(msg: str, *, newline: bool = True) -> None:
    """Write a one-line status to the progress TTY (rank 0 only)."""
    rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
    if str(rank) not in {"", "0"}:
        return
    stream = get_progress_stream()
    if stream is None:
        return
    try:
        end = "\n" if newline else ""
        stream.write(f"\r\033[K{msg}{end}")
        stream.flush()
    except Exception:
        pass


def _open_progress_tty() -> TextIO | None:
    # 1) Concrete pts/path from shell — works without a controlling TTY.
    for path in (
        os.environ.get("LLM4REC_PROGRESS_TTY", "").strip(),
        "/dev/tty",
    ):
        if not path:
            continue
        try:
            fd = os.open(path, os.O_WRONLY)
            return os.fdopen(fd, "w", buffering=1, closefd=True)
        except OSError:
            continue

    # 2) Inherited fd from shell (``exec 3>/dev/tty`` + LLM4REC_PROGRESS_FD=3).
    #    Only useful for the immediate child; elastic workers usually lack it.
    fd_raw = os.environ.get("LLM4REC_PROGRESS_FD", "").strip()
    if fd_raw.isdigit():
        fd = int(fd_raw)
        try:
            if fd > 2:
                os.fstat(fd)
                return os.fdopen(fd, "w", buffering=1, closefd=False)
        except OSError:
            pass

    # 3) Original stderr/stdout if still a real TTY
    for stream in (sys.__stderr__, sys.__stdout__):
        try:
            if stream is not None and stream.isatty():
                return stream
        except Exception:
            continue
    return None


def _patch_transformers_tqdm(task_tqdm_cls: type) -> None:
    try:
        import transformers.utils.logging as hf_logging

        hf_logging._tqdm_cls = task_tqdm_cls  # type: ignore[attr-defined]
        if hasattr(hf_logging, "enable_progress_bar"):
            hf_logging.enable_progress_bar()
        hf_logging._tqdm_active = True  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        import huggingface_hub.utils.tqdm as hub_tqdm

        if hasattr(hub_tqdm, "enable_progress_bars"):
            hub_tqdm.enable_progress_bars()
    except Exception:
        pass


def install_inplace_progress(*, force: bool = False) -> None:
    """Patch tqdm (+ HF wrappers) so bars write to the progress TTY."""
    global _PATCHED, _TTY_STREAM
    if _PATCHED and not force:
        return

    rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
    if str(rank) not in {"", "0"}:
        os.environ["TQDM_DISABLE"] = "1"
        try:
            import transformers.utils.logging as hf_logging

            if hasattr(hf_logging, "disable_progress_bar"):
                hf_logging.disable_progress_bar()
        except Exception:
            pass
        _PATCHED = True
        return

    # Clear any stale TQDM_DISABLE from a previous rank context.
    os.environ.pop("TQDM_DISABLE", None)
    _TTY_STREAM = _open_progress_tty()

    try:
        import tqdm as tqdm_mod
        import tqdm.auto as tqdm_auto
    except Exception:
        _PATCHED = True
        return

    original = tqdm_mod.tqdm

    class _TaskTqdm(original):  # type: ignore[valid-type,misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # Resolve stream at bar creation time (pts may open after install).
            stream = get_progress_stream()
            if stream is not None:
                kwargs["file"] = stream
                kwargs["disable"] = False
            kwargs.setdefault("leave", False)
            kwargs.setdefault("dynamic_ncols", True)
            kwargs.setdefault("mininterval", 0.25)
            # Stage name is a separate line from the shell (`==> name`);
            # keep tqdm desc as the inner progress label only.
            kwargs.setdefault("position", 0)
            super().__init__(*args, **kwargs)

        def close(self) -> None:
            super().close()
            stream = get_progress_stream()
            if stream is not None:
                try:
                    stream.write("\r\033[K")
                    stream.flush()
                except Exception:
                    pass

    tqdm_mod.tqdm = _TaskTqdm  # type: ignore[assignment]
    tqdm_auto.tqdm = _TaskTqdm  # type: ignore[assignment]
    try:
        import tqdm.std as tqdm_std

        tqdm_std.tqdm = _TaskTqdm  # type: ignore[assignment]
    except Exception:
        pass

    _patch_transformers_tqdm(_TaskTqdm)
    _PATCHED = True
