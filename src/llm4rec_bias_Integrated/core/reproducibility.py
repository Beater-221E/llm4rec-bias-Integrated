"""Environment capture and seed management for reproducible runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def set_seed(seed: int) -> None:
    """Seed Python and optional numpy / torch RNGs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _git_info(repo_root: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.check_output(
                args,
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return out.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    return {
        "git_commit": commit,
        "git_dirty": bool(dirty) if dirty is not None else None,
        "git_status_porcelain": dirty if dirty else "",
    }


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _cuda_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda_available": False,
        "cuda_version": None,
        "gpu_count": 0,
        "gpu_names": [],
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["gpu_count"] = int(torch.cuda.device_count())
            info["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    return info


def collect_environment(repo_root: Path | None = None) -> dict[str, Any]:
    """Gather host / library versions for ``environment.json``."""
    root = repo_root or Path.cwd()
    env: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "pytorch_version": _package_version("torch"),
        "transformers_version": _package_version("transformers"),
        "trl_version": _package_version("trl"),
        "peft_version": _package_version("peft"),
        "accelerate_version": _package_version("accelerate"),
        "omegaconf_version": _package_version("omegaconf"),
        "lab_version": _package_version("llm4rec-bias-Integrated") or "0.1.0",
    }
    env.update(_git_info(root))
    env.update(_cuda_info())
    return env


def fingerprint_payload(payload: Any) -> str:
    """Stable SHA256 over a JSON-serializable object."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
