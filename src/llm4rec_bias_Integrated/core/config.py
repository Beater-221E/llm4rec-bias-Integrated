"""YAML config loading, composition, validation, and CLI overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from llm4rec_bias_Integrated.core.exceptions import ConfigurationError
from llm4rec_bias_Integrated.core.paths import base_config_path, configs_dir, project_root

# Named composition slots → subdirectory under configs/
_COMPOSE_SLOTS: dict[str, str] = {
    "dataset": "datasets",
    "model": "models",
    "workflow": "workflows",
    "training": "training",
    "bias": "bias",
    "experiment": "experiments",
    "scale": "scale",
    "hardware": "hardware",
}

# Apply selectors in this order so hardware/scale always win over experiment embeds.
_SELECTOR_ORDER: tuple[str, ...] = (
    "experiment",
    "dataset",
    "model",
    "workflow",
    "training",
    "bias",
    "scale",
    "hardware",
)

# Default profiles when CLI / LLM4REC_COMPOSE omit them.
_DEFAULT_SELECTORS: dict[str, str] = {
    "hardware": "single",
    "scale": "smoke",
}

_REQUIRED_TOP_LEVEL = (
    "experiment",
    "dataset",
    "workflow",
    "model",
    "training",
    "evaluation",
    "tracking",
    "paths",
)


def _as_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise ConfigurationError("Resolved config must be a mapping")
    return container


def parse_overrides(tokens: list[str]) -> dict[str, Any]:
    """Parse Hydra-style ``key=value`` / ``key.nested=value`` tokens."""
    overrides: dict[str, Any] = {}
    for token in tokens:
        if "=" not in token:
            raise ConfigurationError(
                f"Invalid override '{token}'. Expected key=value"
            )
        key, raw = token.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigurationError(f"Empty key in override '{token}'")
        overrides[key] = _parse_scalar(raw.strip())
    return overrides


def _parse_scalar(raw: str) -> Any:
    if raw == "" or raw.lower() in {"null", "none", "~"}:
        return None
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    # list / dict literals via YAML
    if raw.startswith("[") or raw.startswith("{"):
        try:
            return OmegaConf.to_container(OmegaConf.create(raw), resolve=True)
        except Exception as exc:  # noqa: BLE001
            raise ConfigurationError(f"Cannot parse structured override: {raw}") from exc
    try:
        if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
            return int(raw)
        return float(raw)
    except ValueError:
        return raw


def _load_yaml(path: Path) -> DictConfig:
    if not path.is_file():
        raise ConfigurationError(f"Config file not found: {path}")
    return OmegaConf.load(path)  # type: ignore[return-value]


def _resolve_named_file(slot: str, name: str, root: Path) -> Path:
    sub = _COMPOSE_SLOTS[slot]
    # allow either dash or underscore file names
    candidates = [
        root / sub / f"{name}.yaml",
        root / sub / f"{name.replace('-', '_')}.yaml",
        root / sub / f"{name.replace('_', '-')}.yaml",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise ConfigurationError(
        f"No config for {slot}='{name}' under {root / sub}"
    )


def _apply_dot_overrides(cfg: DictConfig, overrides: dict[str, Any]) -> DictConfig:
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=True)
    return cfg


def _lookup_model_alias(models_map: DictConfig, name: str) -> DictConfig | None:
    alias = OmegaConf.select(models_map, name)
    if alias is not None:
        return alias  # type: ignore[return-value]
    target = name.replace("_", "-")
    for key in models_map:
        if str(key).replace("_", "-") == target:
            return models_map[key]  # type: ignore[return-value]
    return None


def _resolve_model_alias(cfg: DictConfig, *, force: bool = False) -> None:
    """Map model.name aliases (e.g. qwen2.5-1b) to checkpoint via models map."""
    models_map = OmegaConf.select(cfg, "models")
    model_cfg = OmegaConf.select(cfg, "model")
    if models_map is None or model_cfg is None:
        return
    name = OmegaConf.select(model_cfg, "name")
    if not name:
        return
    alias = _lookup_model_alias(models_map, str(name))
    if alias is None:
        return
    ckpt = OmegaConf.select(alias, "checkpoint")
    current = OmegaConf.select(model_cfg, "checkpoint")
    if ckpt and (force or current in (None, "", "???")):
        OmegaConf.update(cfg, "model.checkpoint", ckpt, merge=False)


def env_compose_overrides() -> list[str]:
    """Parse ``LLM4REC_COMPOSE`` (space-separated Hydra-style tokens).

    Example::

        export LLM4REC_COMPOSE="hardware=multi scale=full"
    """
    raw = os.environ.get("LLM4REC_COMPOSE", "").strip()
    if not raw:
        return []
    return raw.split()


def apply_hardware_env(cfg: DictConfig | dict[str, Any]) -> None:
    """Apply ``hardware.cuda_visible_devices`` / ``hardware.env`` to ``os.environ``.

    Call before any CUDA init (``require_cuda`` / ``torch.cuda``).
    Explicit shell exports for NCCL_* win via ``setdefault``; CUDA devices from
    the resolved hardware profile always apply so YAML is the source of truth.
    """
    if isinstance(cfg, DictConfig):
        hw = OmegaConf.select(cfg, "hardware")
        hw_dict = OmegaConf.to_container(hw, resolve=True) if hw is not None else {}
    else:
        hw_dict = cfg.get("hardware") or {}
    if not isinstance(hw_dict, dict):
        return

    devices = hw_dict.get("cuda_visible_devices")
    if devices is not None and str(devices).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)

    env_map = hw_dict.get("env") or {}
    if isinstance(env_map, dict):
        for key, value in env_map.items():
            if value is None:
                continue
            os.environ.setdefault(str(key), str(value))


def load_config(
    overrides: list[str] | None = None,
    *,
    config_root: Path | None = None,
    apply_env: bool = True,
) -> DictConfig:
    """Load base config, compose named YAMLs, apply CLI overrides.

    Base file is ``<project_root>/config.yaml``. Named selectors
    (``experiment=…``, ``hardware=…``, ``scale=…``, …) load matching files under
    ``configs/<slot>/``. Defaults: ``hardware=single``, ``scale=smoke``.
    Selector merge order puts ``scale`` then ``hardware`` last so they override
    experiment embeds. Optional ``LLM4REC_COMPOSE`` tokens are prepended by the
    CLI before this function sees ``overrides``.

    ``config_root`` overrides the compose directory (default ``configs/``);
    the base ``config.yaml`` always resolves from the project root unless
    ``config_root`` itself contains a ``config.yaml`` (test override).
    """
    compose_root = config_root or configs_dir()
    base_path = base_config_path()
    if config_root is not None and (config_root / "config.yaml").is_file():
        base_path = config_root / "config.yaml"
    cfg = _load_yaml(base_path)

    parsed = parse_overrides(overrides or [])

    # First pass: apply selectors that choose composition files (fixed order).
    selectors: dict[str, Any] = {
        slot: parsed.pop(slot) for slot in list(parsed) if slot in _COMPOSE_SLOTS
    }
    for slot, default_name in _DEFAULT_SELECTORS.items():
        selectors.setdefault(slot, default_name)

    model_selector_used = "model" in selectors
    for slot in _SELECTOR_ORDER:
        if slot not in selectors:
            continue
        name = selectors[slot]
        if not isinstance(name, str):
            raise ConfigurationError(f"{slot} selector must be a string, got {name!r}")
        if slot == "model":
            # Prefer alias map (qwen2.5-1b); fall back to models/*.yaml file names.
            OmegaConf.update(cfg, "model.name", name, merge=False)
            try:
                named = _load_yaml(_resolve_named_file(slot, name, compose_root))
                cfg = OmegaConf.merge(cfg, named)
            except ConfigurationError:
                pass
            continue
        named = _load_yaml(_resolve_named_file(slot, name, compose_root))
        cfg = OmegaConf.merge(cfg, named)

    # Explicit checkpoint override wins over alias force-refresh.
    explicit_checkpoint = "model.checkpoint" in parsed
    cfg = _apply_dot_overrides(cfg, parsed)
    _resolve_model_alias(cfg, force=model_selector_used and not explicit_checkpoint)

    # If experiment file set nested selectors, compose those too (once)
    for slot, subdir in _COMPOSE_SLOTS.items():
        if slot in {"experiment", "hardware", "scale"}:
            continue
        name = OmegaConf.select(cfg, f"{slot}.name") or OmegaConf.select(cfg, slot)
        if isinstance(name, str) and (compose_root / subdir / f"{name}.yaml").exists():
            # Only merge if the named file adds fields not already fully present
            # Skip if already merged via selector
            if slot not in selectors:
                named_path = _resolve_named_file(slot, name, compose_root)
                named = _load_yaml(named_path)
                cfg = OmegaConf.merge(cfg, named)

    # Re-apply late profiles so nested experiment merges cannot clobber them.
    for slot in ("scale", "hardware"):
        name = selectors.get(slot)
        if isinstance(name, str):
            cfg = OmegaConf.merge(
                cfg, _load_yaml(_resolve_named_file(slot, name, compose_root))
            )

    # CLI dot-overrides still win over composed profiles.
    if parsed:
        cfg = _apply_dot_overrides(cfg, parsed)

    _resolve_model_alias(cfg, force=model_selector_used and not explicit_checkpoint)
    OmegaConf.resolve(cfg)
    if apply_env:
        apply_hardware_env(cfg)
    return cfg

def validate_config(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Fail-fast schema checks for Phase 1 (extended in later phases)."""
    data = _as_dict(cfg) if not isinstance(cfg, dict) else cfg
    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in data]
    if missing:
        raise ConfigurationError(f"Missing top-level keys: {missing}")

    experiment = data["experiment"]
    if not isinstance(experiment, dict) or not experiment.get("name"):
        raise ConfigurationError("experiment.name is required")
    seed = experiment.get("seed", data.get("seed"))
    if seed is None:
        raise ConfigurationError("experiment.seed (or seed) is required")
    if not isinstance(seed, int):
        raise ConfigurationError(f"seed must be int, got {type(seed).__name__}")

    dataset = data["dataset"]
    if not isinstance(dataset, dict) or not dataset.get("name"):
        raise ConfigurationError("dataset.name is required")

    workflow = data["workflow"]
    if not isinstance(workflow, dict) or not workflow.get("name"):
        raise ConfigurationError("workflow.name is required")

    model = data["model"]
    if not isinstance(model, dict) or not model.get("name"):
        raise ConfigurationError("model.name is required")
    if not model.get("checkpoint"):
        raise ConfigurationError(
            "model.checkpoint is required (resolve via models alias map if needed)"
        )

    training = data["training"]
    if not isinstance(training, dict):
        raise ConfigurationError("training must be a mapping")
    stages = training.get("stages")
    if not stages or not isinstance(stages, list):
        raise ConfigurationError("training.stages must be a non-empty list")

    grpo = data.get("grpo") or {}
    if isinstance(grpo, dict) and grpo:
        weights = grpo.get("reward_weights") or {}
        if weights and all(float(v) == 0.0 for v in weights.values()):
            raise ConfigurationError("grpo.reward_weights cannot be all zeros")
        n_gen = grpo.get("num_generations")
        if n_gen is not None and int(n_gen) < 2:
            raise ConfigurationError("grpo.num_generations must be >= 2")

    peft = data.get("peft") or {}
    model_name = str(model.get("name", "")).lower()
    if "7b" in model_name and isinstance(peft, dict) and peft.get("enabled") is False:
        raise ConfigurationError(
            "Full-parameter training for 7B models is disabled by default; "
            "enable peft or override explicitly after acknowledging VRAM risk"
        )

    return data


def save_resolved_config(cfg: DictConfig | dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(cfg, dict):
        OmegaConf.save(OmegaConf.create(cfg), path)
    else:
        OmegaConf.save(cfg, path)


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    return _as_dict(cfg)


def default_repo_root() -> Path:
    return project_root()
