"""YAML config loading: three routes × training stages."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from llm4rec.core.exceptions import ConfigurationError
from llm4rec.core.paths import base_config_path, project_root

_ROUTES = ("grpo4rec", "minionerec", "mllm4rec")

# Stage keys under each letter/SID route (merged into active runtime config)
_LETTER_STAGES = ("prepare", "sft", "grpo", "evaluate", "analyze")

_COMPOSE_SLOTS = {
    "dataset",
    "model",
    "workflow",
    "training",
    "bias",
    "reward",
    "evaluation",
    "experiment",
    "scale",
    "hardware",
}

_DEFAULT_SELECTORS = {
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
    overrides: dict[str, Any] = {}
    for token in tokens:
        if "=" not in token:
            raise ConfigurationError(f"Invalid override '{token}'. Expected key=value")
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


def _apply_dot_overrides(cfg: DictConfig, overrides: dict[str, Any]) -> DictConfig:
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=True)
    return cfg


def _node_as_dict(node: Any) -> dict[str, Any]:
    if node is None:
        return {}
    if isinstance(node, DictConfig):
        out = OmegaConf.to_container(node, resolve=False)
        return dict(out) if isinstance(out, dict) else {}
    if isinstance(node, dict):
        return dict(node)
    return {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(overlay)))


def _lookup_model_alias(models_map: DictConfig | dict[str, Any] | None, name: str) -> Any:
    if models_map is None:
        return None
    if isinstance(models_map, DictConfig):
        alias = OmegaConf.select(models_map, name)
        if alias is not None:
            return alias
        target = name.replace("_", "-")
        for key in models_map:
            if str(key).replace("_", "-") == target:
                return models_map[key]
        return None
    return models_map.get(name) or models_map.get(name.replace("_", "-"))


def _resolve_model_alias(cfg: DictConfig, *, force: bool = False) -> None:
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
    ckpt = OmegaConf.select(alias, "checkpoint") if isinstance(alias, DictConfig) else alias.get("checkpoint")
    current = OmegaConf.select(model_cfg, "checkpoint")
    if ckpt and (force or current in (None, "", "???")):
        OmegaConf.update(cfg, "model.checkpoint", ckpt, merge=False)


def env_compose_overrides() -> list[str]:
    raw = os.environ.get("LLM4REC_COMPOSE", "").strip()
    if not raw:
        return []
    return raw.split()


def apply_hardware_env(cfg: DictConfig | dict[str, Any]) -> None:
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


def _experiment_spec(root: DictConfig, name: str) -> dict[str, Any]:
    node = OmegaConf.select(root, f"experiments.{name}")
    if node is None:
        raise ConfigurationError(f"Unknown experiment='{name}' (see config.yaml experiments:)")
    return _node_as_dict(node)


def _flatten_route(
    root: DictConfig,
    route: str,
    *,
    stages: list[str],
    scale: str,
    extra_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a flat runtime config from ``<route>.{shared,stages,scale}``."""
    route_node = OmegaConf.select(root, route)
    if route_node is None:
        raise ConfigurationError(f"Unknown workflow/route '{route}'")
    route_cfg = _node_as_dict(route_node)

    # Shared route fields (not stage / scale keys)
    skip = set(_LETTER_STAGES) | {"smoke", "full", "data", "retriever", "ranker", "name"}
    active: dict[str, Any] = {
        k: v for k, v in route_cfg.items() if k not in skip and not k.startswith("_")
    }

    # Merge each requested stage block
    for stage in stages:
        if stage in {"report", "prepare"} and stage not in route_cfg:
            continue
        block = _node_as_dict(route_cfg.get(stage))
        if not block:
            continue
        # Stage blocks already wrap training:/grpo:/evaluation: etc.
        active = _deep_merge(active, block)

    # Scale preset on the route (smoke / full)
    preset = _node_as_dict(route_cfg.get(scale))
    if preset:
        # preset may nest stage overrides: sft: {training: ...}
        flat_preset: dict[str, Any] = {}
        for key, value in preset.items():
            if key in _LETTER_STAGES or key in {"retriever", "ranker", "data"}:
                flat_preset = _deep_merge(flat_preset, _node_as_dict(value))
            else:
                flat_preset[key] = value
        active = _deep_merge(active, flat_preset)

    if extra_overrides:
        active = _deep_merge(active, extra_overrides)

    # Ensure training.stages lists the pipeline to run
    training = dict(active.get("training") or {})
    training["stages"] = list(stages)
    active["training"] = training

    # Ensure workflow.name
    wf = dict(active.get("workflow") or {})
    wf.setdefault("name", route)
    active["workflow"] = wf

    return active


def _apply_hardware(root: DictConfig, active: dict[str, Any], hardware: str) -> dict[str, Any]:
    node = OmegaConf.select(root, f"hardware_profiles.{hardware}")
    if node is None:
        # allow legacy key hardware.single
        node = OmegaConf.select(root, f"hardware.{hardware}")
    if node is None:
        raise ConfigurationError(f"Unknown hardware='{hardware}'")
    return _deep_merge(active, _node_as_dict(node))


def load_profile(slot: str, name: str) -> dict[str, Any]:
    """Load a route stage fragment for MLLM CLI.

    Accepts names like ``mllm4rec_retriever``, ``mllm4rec.retriever``,
    ``retriever``, ``mllm4rec_ml100k``, ``mllm4rec.data.ml100k``.
    """
    root = _load_yaml(base_config_path())
    text = name.strip()
    # Normalize aliases
    aliases = {
        "mllm4rec_retriever": ("mllm4rec", "retriever"),
        "mllm4rec_ranker": ("mllm4rec", "ranker"),
        "mllm4rec_ml100k": ("mllm4rec", "data", "ml100k"),
        "mllm4rec_ml1m": ("mllm4rec", "data", "ml1m"),
    }
    if text in aliases:
        path = ".".join(aliases[text])
        node = OmegaConf.select(root, path)
        if node is not None:
            return _node_as_dict(node)

    # dotted path: mllm4rec.retriever / mllm4rec.data.ml100k
    if "." in text:
        node = OmegaConf.select(root, text)
        if node is not None:
            return _node_as_dict(node)

    # bare stage under mllm4rec
    for path in (f"mllm4rec.{text}", f"mllm4rec.data.{text}", text):
        node = OmegaConf.select(root, path)
        if node is not None:
            return _node_as_dict(node)

    raise ConfigurationError(
        f"Unknown profile '{name}'. Use e.g. mllm4rec_retriever / mllm4rec.data.ml100k"
    )


def load_config(
    overrides: list[str] | None = None,
    *,
    config_root: Path | None = None,
    apply_env: bool = True,
) -> DictConfig:
    """Compose runtime config from route × stages × scale × hardware."""
    base_path = base_config_path()
    if config_root is not None and (config_root / "config.yaml").is_file():
        base_path = config_root / "config.yaml"
    root = _load_yaml(base_path)

    parsed = parse_overrides(overrides or [])
    cli_selectors = {k: parsed.pop(k) for k in list(parsed) if k in _COMPOSE_SLOTS}
    selectors = {**_DEFAULT_SELECTORS, **cli_selectors}

    # Resolve experiment → workflow/scale/stages
    exp_name = selectors.get("experiment")
    exp_spec: dict[str, Any] = {}
    if isinstance(exp_name, str):
        exp_spec = _experiment_spec(root, exp_name)
        if "workflow" not in cli_selectors and exp_spec.get("workflow"):
            selectors["workflow"] = exp_spec["workflow"]
        if "scale" not in cli_selectors and exp_spec.get("scale"):
            selectors["scale"] = exp_spec["scale"]

    workflow = selectors.get("workflow")
    if not isinstance(workflow, str):
        workflow = "grpo4rec"
    if workflow not in _ROUTES:
        raise ConfigurationError(f"workflow must be one of {_ROUTES}, got {workflow!r}")

    scale = str(selectors.get("scale") or "smoke")
    hardware = str(selectors.get("hardware") or "single")
    stages = list(exp_spec.get("stages") or ["sft", "evaluate"])
    extra = dict(exp_spec.get("overrides") or {})

    # Allow workflow= without experiment: use a sensible default stage set
    if not isinstance(exp_name, str):
        if workflow == "mllm4rec":
            stages = ["retriever", "ranker"]
        else:
            stages = ["sft", "grpo", "evaluate"]

    active = _flatten_route(
        root,
        workflow,
        stages=[s for s in stages if s not in {"report"}],
        scale=scale,
        extra_overrides=extra,
    )
    active = _apply_hardware(root, active, hardware)

    # Global bits
    for key in ("paths", "models", "tracking"):
        if key not in active and OmegaConf.select(root, key) is not None:
            active[key] = _node_as_dict(OmegaConf.select(root, key))
    if "seed" not in active:
        active["seed"] = OmegaConf.select(root, "seed") or 42

    # Experiment metadata
    active["experiment"] = {
        "name": str(exp_name or f"{workflow}_{scale}"),
        "seed": int(active.get("seed") or 42),
    }
    active["scale"] = {"name": scale}

    active.setdefault("evaluation", {"top_k": [1, 5, 10], "use_upstream_eval": True})
    active.setdefault("tracking", _node_as_dict(OmegaConf.select(root, "tracking")))
    active.setdefault("paths", _node_as_dict(OmegaConf.select(root, "paths")))

    # Optional reward / evaluation presets
    reward_name = cli_selectors.get("reward")
    if isinstance(reward_name, str):
        preset = OmegaConf.select(root, f"reward_presets.{reward_name}")
        if preset is not None:
            active = _deep_merge(active, _node_as_dict(preset))
    eval_name = cli_selectors.get("evaluation")
    if isinstance(eval_name, str):
        preset = OmegaConf.select(root, f"evaluation_presets.{eval_name}")
        if preset is not None:
            active = _deep_merge(active, _node_as_dict(preset))

    cfg = OmegaConf.create(active)

    # model= / dataset= selectors
    model_selector_used = "model" in cli_selectors
    if model_selector_used:
        OmegaConf.update(cfg, "model.name", cli_selectors["model"], merge=False)

    if "dataset" in cli_selectors and isinstance(cli_selectors["dataset"], str):
        ds_name = str(cli_selectors["dataset"])
        alias = {"ml1m": "movielens_1m", "ml100k": "movielens_100k"}.get(ds_name, ds_name)
        OmegaConf.update(cfg, "dataset.name", alias, merge=False)

    explicit_checkpoint = "model.checkpoint" in parsed
    cfg = _apply_dot_overrides(cfg, parsed)
    _resolve_model_alias(cfg, force=model_selector_used and not explicit_checkpoint)

    if parsed:
        cfg = _apply_dot_overrides(cfg, parsed)
    _resolve_model_alias(cfg, force=model_selector_used and not explicit_checkpoint)

    OmegaConf.resolve(cfg)
    if apply_env:
        apply_hardware_env(cfg)
    return cfg


def validate_config(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
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
