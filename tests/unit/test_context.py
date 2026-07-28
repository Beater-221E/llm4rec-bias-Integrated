"""Tests for experiment context and reproducibility helpers."""

from __future__ import annotations

from pathlib import Path

from llm4rec_bias_Integrated.core.config import load_config
from llm4rec_bias_Integrated.core.context import create_context
from llm4rec_bias_Integrated.core.reproducibility import collect_environment, fingerprint_payload


def test_create_context_writes_resolved_artifacts(tmp_path: Path) -> None:
    cfg = load_config(["experiment=smoke_test"])
    ctx = create_context(
        cfg,
        cli_overrides=["experiment=smoke_test"],
        create_run_dir=True,
        run_dir=tmp_path / "run",
    )
    assert (tmp_path / "run" / "resolved_config.yaml").is_file()
    assert (tmp_path / "run" / "environment.json").is_file()
    assert (tmp_path / "run" / "metrics.jsonl").is_file()
    assert ctx.seed == 42
    assert "movielens_100k" in ctx.experiment_id


def test_fingerprint_stable() -> None:
    assert fingerprint_payload({"a": 1, "b": [2, 3]}) == fingerprint_payload(
        {"b": [2, 3], "a": 1}
    )


def test_collect_environment_has_python() -> None:
    env = collect_environment()
    assert "python_version" in env
    assert "hostname" in env
