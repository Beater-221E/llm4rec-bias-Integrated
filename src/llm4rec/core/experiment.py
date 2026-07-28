"""Experiment lifecycle: run dirs, config snapshot, git SHA, resume."""

from __future__ import annotations

import secrets
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from llm4rec.core.checkpoint import CheckpointManager
from llm4rec.core.config import save_resolved_config, validate_config
from llm4rec.core.context import (
    ExperimentContext,
    build_run_dir,
    create_context,
    dry_validate,
)
from llm4rec.core.exceptions import ConfigurationError
from llm4rec.core.logging import build_logger
from llm4rec.core.paths import ensure_dir, project_root
from llm4rec.core.reproducibility import collect_environment, set_seed, write_json

__all__ = [
    "ExperimentContext",
    "ExperimentManager",
    "ExperimentRecord",
    "build_run_dir",
    "create_context",
    "dry_validate",
]


def _git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


@dataclass
class ExperimentRecord:
    """Artifacts written under a single runs/ directory."""

    run_dir: Path
    config: dict[str, Any]
    experiment_id: str
    seed: int
    git_commit: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class ExperimentManager:
    """Unified experiment bookkeeping for reproducible LLM4Rec runs.

    Persists under ``runs/``:

    - ``config.yaml`` / ``resolved_config.yaml``
    - ``checkpoints/``
    - logs (via tracking backends)
    - ``metrics.json``
    - ``environment.json`` (includes git commit)
    """

    def __init__(self, config: dict[str, Any] | DictConfig) -> None:
        if isinstance(config, DictConfig):
            self.config = validate_config(config)
        else:
            self.config = dict(config)
        self.seed = int(self.config["experiment"]["seed"])
        self._context: ExperimentContext | None = None
        self._record: ExperimentRecord | None = None

    @classmethod
    def from_overrides(
        cls,
        overrides: list[str] | None = None,
        *,
        create_run_dir: bool = True,
    ) -> tuple["ExperimentManager", ExperimentContext]:
        from llm4rec.core.config import load_config

        cfg = load_config(overrides or [])
        mgr = cls(cfg)
        ctx = mgr.start(
            create_run_dir=create_run_dir,
            cli_overrides=list(overrides or []),
        )
        return mgr, ctx

    def start(
        self,
        *,
        create_run_dir: bool = True,
        run_dir: Path | None = None,
        cli_overrides: list[str] | None = None,
        resume_dir: Path | None = None,
    ) -> ExperimentContext:
        set_seed(self.seed)
        if resume_dir is not None:
            out = ensure_dir(Path(resume_dir))
        elif run_dir is not None:
            out = ensure_dir(run_dir)
        elif create_run_dir:
            out = build_run_dir(self.config)
        else:
            out = Path(".")

        experiment_id = (
            f"{self.config['experiment']['name']}__{self.config['dataset']['name']}__"
            f"{self.config['workflow']['name']}__{self.config['model']['name']}__"
            f"seed{self.seed}"
        )
        git_sha = _git_commit(project_root())
        if create_run_dir or run_dir is not None or resume_dir is not None:
            save_resolved_config(self.config, out / "resolved_config.yaml")
            save_resolved_config(self.config, out / "config.yaml")
            env = collect_environment(project_root())
            env["seed"] = self.seed
            env["experiment_id"] = experiment_id
            env["cli_overrides"] = list(cli_overrides or [])
            env["git_commit"] = git_sha
            env["dataset"] = self.config["dataset"]["name"]
            env["model"] = self.config["model"]["name"]
            env["workflow"] = self.config["workflow"]["name"]
            env["reward"] = (self.config.get("reward") or {}).get("name")
            env["evaluation"] = (self.config.get("evaluation") or {}).get("name")
            write_json(out / "environment.json", env)
        else:
            env = {"git_commit": git_sha}

        logger = build_logger(
            self.config.get("tracking", {}),
            out if create_run_dir or run_dir or resume_dir else None,
        )
        ctx = ExperimentContext(
            config=self.config,
            run_dir=out,
            experiment_id=experiment_id,
            seed=self.seed,
            logger=logger,
            cli_overrides=list(cli_overrides or []),
            environment=env,
        )
        self._context = ctx
        self._record = ExperimentRecord(
            run_dir=out,
            config=self.config,
            experiment_id=experiment_id,
            seed=self.seed,
            git_commit=git_sha,
        )
        return ctx

    @property
    def context(self) -> ExperimentContext:
        if self._context is None:
            raise ConfigurationError("ExperimentManager.start() has not been called")
        return self._context

    def checkpoint_manager(self) -> CheckpointManager:
        return CheckpointManager(self.context.run_dir / "checkpoints")

    def save_metrics(self, metrics: dict[str, Any], *, name: str = "metrics.json") -> Path:
        path = self.context.run_dir / name
        write_json(path, metrics)
        if self._record is not None:
            self._record.metrics.update(metrics)
        return path

    def resume(self, run_dir: Path) -> ExperimentContext:
        """Attach to an existing run directory (resume training / eval)."""
        cfg_path = Path(run_dir) / "resolved_config.yaml"
        if not cfg_path.is_file():
            raise ConfigurationError(f"Cannot resume: missing {cfg_path}")
        from omegaconf import OmegaConf

        loaded = OmegaConf.load(cfg_path)
        self.config = validate_config(loaded)
        self.seed = int(self.config["experiment"]["seed"])
        return self.start(resume_dir=Path(run_dir), create_run_dir=False)
