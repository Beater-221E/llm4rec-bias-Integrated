"""Experiment context: resolved config, run directory, seed, loggers."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from llm4rec.core.config import config_to_dict, save_resolved_config, validate_config
from llm4rec.core.exceptions import ConfigurationError
from llm4rec.core.paths import ensure_dir, project_root, runs_dir
from llm4rec.core.reproducibility import collect_environment, set_seed, write_json
from llm4rec.tracking.logger import ExperimentLogger, build_logger


def _short_id(n: int = 6) -> str:
    return secrets.token_hex(n // 2 + 1)[:n]


@dataclass
class ExperimentContext:
    """Runtime handle for a single experiment run."""

    config: dict[str, Any]
    run_dir: Path
    experiment_id: str
    seed: int
    logger: ExperimentLogger
    cli_overrides: list[str] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def dataset_name(self) -> str:
        return str(self.config["dataset"]["name"])

    @property
    def workflow_name(self) -> str:
        return str(self.config["workflow"]["name"])

    @property
    def model_name(self) -> str:
        return str(self.config["model"]["name"])

    def summary_lines(self) -> list[str]:
        train = self.config.get("training", {})
        peft = self.config.get("peft", {})
        return [
            f"Experiment ID     : {self.experiment_id}",
            f"Dataset           : {self.dataset_name}",
            f"Workflow          : {self.workflow_name}",
            f"Model             : {self.model_name}",
            f"Checkpoint        : {self.config['model'].get('checkpoint')}",
            f"Stages            : {train.get('stages')}",
            f"Seed              : {self.seed}",
            f"PEFT              : {peft.get('enabled')} ({peft.get('method')})",
            f"Precision         : {self.config['model'].get('dtype')}",
            f"Output directory  : {self.run_dir}",
        ]


def build_run_dir(config: dict[str, Any], *, base: Path | None = None) -> Path:
    """Create ``runs/<dataset>/<workflow>/<model>/seed_<seed>/<ts>_<id>/``."""
    dataset = str(config["dataset"]["name"])
    workflow = str(config["workflow"]["name"])
    model = str(config["model"]["name"]).replace("/", "_")
    seed = int(config["experiment"]["seed"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{stamp}_{_short_id()}"
    root = base or runs_dir()
    path = root / dataset / workflow / model / f"seed_{seed}" / run_id
    if path.exists():
        raise ConfigurationError(f"Output directory already exists: {path}")
    return ensure_dir(path)


def create_context(
    cfg: DictConfig,
    *,
    cli_overrides: list[str] | None = None,
    create_run_dir: bool = True,
    run_dir: Path | None = None,
) -> ExperimentContext:
    """Validate config, optionally allocate a run directory, wire loggers."""
    config = validate_config(cfg)
    seed = int(config["experiment"]["seed"])
    set_seed(seed)

    if run_dir is not None:
        out = ensure_dir(run_dir)
    elif create_run_dir:
        out = build_run_dir(config)
    else:
        out = Path(".")

    experiment_id = (
        f"{config['experiment']['name']}__{config['dataset']['name']}__"
        f"{config['workflow']['name']}__{config['model']['name']}__seed{seed}"
    )
    if create_run_dir or run_dir is not None:
        save_resolved_config(config, out / "resolved_config.yaml")
        env = collect_environment(project_root())
        env["seed"] = seed
        env["experiment_id"] = experiment_id
        env["cli_overrides"] = list(cli_overrides or [])
        write_json(out / "environment.json", env)
    else:
        env = {}

    logger = build_logger(config.get("tracking", {}), out if create_run_dir or run_dir else None)
    return ExperimentContext(
        config=config,
        run_dir=out,
        experiment_id=experiment_id,
        seed=seed,
        logger=logger,
        cli_overrides=list(cli_overrides or []),
        environment=env,
    )


def dry_validate(cfg: DictConfig) -> dict[str, Any]:
    """Validate without creating a run directory."""
    return validate_config(cfg)
