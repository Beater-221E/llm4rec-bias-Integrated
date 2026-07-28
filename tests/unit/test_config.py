"""Unit tests for config composition and validation (Phase 1)."""

from __future__ import annotations

import os

import pytest
from omegaconf import OmegaConf

from llm4rec_bias_Integrated.core.config import load_config, parse_overrides, validate_config
from llm4rec_bias_Integrated.core.exceptions import ConfigurationError
from llm4rec_bias_Integrated.core.registry import Registry


def test_parse_overrides_scalars_and_lists() -> None:
    parsed = parse_overrides(
        ["seed=7", "tracking.wandb=false", "bias.probes=[popularity,position]"]
    )
    assert parsed["seed"] == 7
    assert parsed["tracking.wandb"] is False
    assert parsed["bias.probes"] == ["popularity", "position"]


def test_smoke_test_config_validates() -> None:
    cfg = load_config(["experiment=smoke_test"], apply_env=False)
    data = validate_config(cfg)
    assert data["experiment"]["name"] == "smoke_test"
    assert data["dataset"]["name"] == "movielens_100k"
    assert data["workflow"]["name"] == "grpo4rec"
    assert data["model"]["checkpoint"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert data["training"]["stages"][0] == "sft"
    assert data["hardware"]["name"] == "single"
    assert data["training"]["distributed"] == "single"
    assert data["training"]["auto_launch_multi_gpu"] is False
    assert data["scale"]["name"] == "smoke"


def test_hardware_multi_overrides_distributed() -> None:
    cfg = load_config(
        ["experiment=smoke_grpo", "hardware=multi"],
        apply_env=False,
    )
    data = validate_config(cfg)
    assert data["hardware"]["name"] == "multi"
    assert data["hardware"]["cuda_visible_devices"] == "0,1,2,3"
    assert data["training"]["distributed"] == "auto"
    assert data["training"]["auto_launch_multi_gpu"] is True
    assert data["dataset"]["train_limit"] == 64  # smoke experiment limit kept


def test_scale_full_clears_smoke_limits() -> None:
    cfg = load_config(
        ["experiment=smoke_grpo", "hardware=multi", "scale=full"],
        apply_env=False,
    )
    data = validate_config(cfg)
    assert data["scale"]["name"] == "full"
    assert data["dataset"]["train_limit"] is None
    assert data["dataset"]["eval_limit"] is None
    assert data["training"]["max_steps"] is None
    assert data["grpo"]["max_steps"] == 100
    assert data["training"]["distributed"] == "auto"


def test_apply_hardware_env_sets_cuda_devices(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    cfg = load_config(["experiment=smoke_test", "hardware=multi"], apply_env=True)
    assert cfg.hardware.name == "multi"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert os.environ["NCCL_P2P_DISABLE"] == "1"


def test_env_compose_overrides(monkeypatch) -> None:
    monkeypatch.setenv("LLM4REC_COMPOSE", "hardware=multi scale=full")
    from llm4rec_bias_Integrated.core.config import env_compose_overrides

    assert env_compose_overrides() == ["hardware=multi", "scale=full"]


def test_model_alias_1b_maps_to_1_5b_checkpoint() -> None:
    cfg = load_config(["experiment=smoke_test", "model=qwen2.5-1b"], apply_env=False)
    data = validate_config(cfg)
    assert data["model"]["name"] == "qwen2.5-1b"
    assert data["model"]["checkpoint"] == "Qwen/Qwen2.5-1.5B-Instruct"


def test_zero_reward_weights_fail() -> None:
    cfg = load_config(
        [
            "experiment=smoke_test",
            "grpo.reward_weights.exact_match=0",
            "grpo.reward_weights.format_validity=0",
            "grpo.reward_weights.popularity_penalty=0",
        ],
        apply_env=False,
    )
    # smoke_test only sets three weights; ensure all present are zero
    with pytest.raises(ConfigurationError, match="all zeros"):
        validate_config(cfg)


def test_group_size_must_be_at_least_two() -> None:
    cfg = load_config(
        ["experiment=smoke_test", "grpo.num_generations=1"],
        apply_env=False,
    )
    with pytest.raises(ConfigurationError, match="num_generations"):
        validate_config(cfg)


def test_registry_rejects_duplicates() -> None:
    reg: Registry[str] = Registry("toy")

    @reg.register("foo")
    def _factory() -> str:
        return "foo"

    with pytest.raises(ConfigurationError):

        @reg.register("foo")
        def _again() -> str:
            return "bar"


def test_missing_experiment_name_fails() -> None:
    cfg = OmegaConf.create(
        {
            "experiment": {"seed": 1},
            "dataset": {"name": "x"},
            "workflow": {"name": "y"},
            "model": {"name": "z", "checkpoint": "c"},
            "training": {"stages": ["sft"]},
            "evaluation": {},
            "tracking": {},
            "paths": {},
        }
    )
    with pytest.raises(ConfigurationError, match="experiment.name"):
        validate_config(cfg)
