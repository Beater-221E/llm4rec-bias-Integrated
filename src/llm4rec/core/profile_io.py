"""Load a YAML file or a named profile from the root ``config.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llm4rec.core.config import load_profile
from llm4rec.core.exceptions import ConfigurationError
from llm4rec.core.paths import base_config_path, project_root


def load_yaml_or_profile(
    path_or_name: str | Path | None,
    *,
    default_slots: tuple[str, ...] = ("training", "mllm_dataset", "dataset"),
) -> dict[str, Any]:
    """Resolve ``--config``.

    Accepts:

    - a filesystem path to a ``.yaml`` file
    - a profile name defined under ``profiles.*`` in ``config.yaml``
      (e.g. ``mllm4rec_retriever``, ``mllm4rec_ml100k``)
    """
    if not path_or_name:
        return {}
    text = str(path_or_name).strip()
    path = Path(text)
    if not path.is_absolute():
        cand = project_root() / path
        if cand.is_file():
            path = cand
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ConfigurationError(f"Config must be a mapping: {path}")
        return data

    # Profile name (no path / missing file)
    name = path.name if text.endswith((".yaml", ".yml")) else text
    # strip directory prefixes like configs/training/
    name = Path(name).stem
    last_err: Exception | None = None
    for slot in default_slots:
        try:
            return load_profile(slot, name)
        except ConfigurationError as exc:
            last_err = exc
            continue
    # Also try loading any matching key under profiles.*
    from omegaconf import OmegaConf

    root = OmegaConf.load(base_config_path())
    profiles = OmegaConf.select(root, "profiles")
    if profiles is not None:
        for slot in profiles:
            node = OmegaConf.select(profiles, f"{slot}.{name}")
            if node is None:
                node = OmegaConf.select(profiles, f"{slot}.{name.replace('-', '_')}")
            if node is not None:
                out = OmegaConf.to_container(node, resolve=True)
                if isinstance(out, dict):
                    return out
    raise ConfigurationError(
        f"Cannot resolve config '{path_or_name}' as file or profile. "
        f"Last error: {last_err}"
    )
