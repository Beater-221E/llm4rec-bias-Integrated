"""Path helpers for project roots and run directories."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[3]


def configs_dir() -> Path:
    """Compose-slot YAMLs (``hardware/``, ``experiments/``, …) under ``configs/``."""
    return project_root() / "configs"


def base_config_path() -> Path:
    """Global defaults: ``<project_root>/config.yaml``."""
    return project_root() / "config.yaml"


def data_dir() -> Path:
    return project_root() / "data"


def runs_dir() -> Path:
    return project_root() / "runs"


def reports_dir() -> Path:
    return project_root() / "reports"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
