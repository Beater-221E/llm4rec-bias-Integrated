"""Run bias probes against a checkpoint / run directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from rich.console import Console
from rich.table import Table

from llm4rec.core.exceptions import ConfigurationError, MissingArtifactError
from llm4rec.core.paths import project_root
from llm4rec.core.reproducibility import write_json
from llm4rec.core.schemas import ProbeResult
from llm4rec.components.dataset._impl.registry import build_dataset
from llm4rec.components.model._impl.base import require_cuda
from llm4rec.components.model._impl.loader import load_model_bundle
from llm4rec.components.evaluation.probes.registry import build_probes_from_config
from llm4rec.workflows.grpo4rec import task_spec_from_config

console = Console()


def resolve_checkpoint_paths(
    run_dir: Path,
    *,
    checkpoint_stage: str | None = None,
    adapter_path: str | None = None,
    sft_adapter_path: str | None = None,
) -> tuple[str | None, str | None]:
    """Return (adapter_path, sft_adapter_path) for load_model_bundle.

    Preference: explicit overrides > grpo(+sft merge) > sft.
    """
    if adapter_path:
        return adapter_path, sft_adapter_path

    sft_final = run_dir / "checkpoints" / "sft" / "final"
    grpo_final = run_dir / "checkpoints" / "grpo" / "final"
    stage = (checkpoint_stage or "").strip().lower() or None

    if stage == "sft":
        if not sft_final.is_dir():
            raise MissingArtifactError(f"missing SFT adapter at {sft_final}")
        return str(sft_final), None
    if stage == "grpo":
        if not grpo_final.is_dir():
            raise MissingArtifactError(f"missing GRPO adapter at {grpo_final}")
        sft = str(sft_final) if sft_final.is_dir() else None
        return str(grpo_final), sft

    # Auto
    if grpo_final.is_dir():
        sft = str(sft_final) if sft_final.is_dir() else None
        return str(grpo_final), sft
    if sft_final.is_dir():
        return str(sft_final), None
    raise MissingArtifactError(
        f"No checkpoints/sft/final or checkpoints/grpo/final under {run_dir}"
    )


def load_run_config(run_dir: Path, fallback_cfg: dict[str, Any]) -> dict[str, Any]:
    resolved = run_dir / "resolved_config.yaml"
    if resolved.is_file():
        with resolved.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict) and data:
            # Overlay bias / evaluation knobs from current CLI config
            merged = dict(data)
            if "bias" in fallback_cfg:
                merged["bias"] = fallback_cfg["bias"]
            if "evaluation" in fallback_cfg:
                merged.setdefault("evaluation", {})
                # keep run model/dataset; allow max_examples overrides via bias
            return merged
    return fallback_cfg


def _build_dataset(cfg: dict[str, Any]):
    ds_cfg = cfg["dataset"]
    root = Path((cfg.get("paths") or {}).get("data_root", "data"))
    if not root.is_absolute():
        root = project_root() / root
    return build_dataset(
        str(ds_cfg["name"]),
        data_root=root,
        rating_threshold=float(ds_cfg.get("rating_threshold", 4.0)),
        split=str(ds_cfg.get("split", "leave_one_out")),
        history_max_length=int(ds_cfg.get("history_max_length", 20)),
        candidate_size=int(ds_cfg.get("candidate_size", 10)),
        negative_sampling=str(ds_cfg.get("negative_sampling", "uniform")),
        target_position=str(ds_cfg.get("target_position", "random")),
        framing=str(ds_cfg.get("framing", "neutral")),
        min_user_interactions=int(ds_cfg.get("min_user_interactions", 5)),
        seed=int(cfg["experiment"]["seed"]),
        train_limit=ds_cfg.get("train_limit"),
        eval_limit=ds_cfg.get("eval_limit"),
    )


def run_probes(
    *,
    cfg: dict[str, Any],
    run_dir: Path,
    checkpoint_stage: str | None = None,
    adapter_path: str | None = None,
    sft_adapter_path: str | None = None,
    split: str = "test",
) -> dict[str, Any]:
    require_cuda()
    run_cfg = load_run_config(run_dir, cfg)
    bias_cfg = dict(run_cfg.get("bias") or cfg.get("bias") or {})
    # Prefer CLI bias overrides already merged into cfg
    if cfg.get("bias"):
        bias_cfg = dict(cfg["bias"])

    adapter, sft_adapter = resolve_checkpoint_paths(
        run_dir,
        checkpoint_stage=checkpoint_stage,
        adapter_path=adapter_path,
        sft_adapter_path=sft_adapter_path,
    )

    dataset = _build_dataset(run_cfg)
    dataset.download()
    dataset.preprocess()
    examples = dataset.build_examples(split, task_spec_from_config(run_cfg))
    max_examples = bias_cfg.get("max_examples")
    if max_examples is None:
        max_examples = (run_cfg.get("evaluation") or {}).get("max_examples")
    if max_examples is not None:
        examples = examples[: int(max_examples)]
    if not examples:
        raise ConfigurationError("No examples available for probes")

    # Ensure history_titles present (legacy examples without the field)
    for ex in examples:
        if "history_titles" not in ex.features:
            raise ConfigurationError(
                "examples missing features.history_titles — re-run prepare/build "
                "with current lab version"
            )

    tok, model, _ = load_model_bundle(
        run_cfg["model"],
        peft_cfg=None,
        adapter_path=adapter,
        sft_adapter_path=sft_adapter,
        for_training=False,
    )
    device = next(model.parameters()).device
    model.eval()

    probes = build_probes_from_config(bias_cfg)
    out_dir = run_dir / "eval" / "probes"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[ProbeResult] = []
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "adapter_path": adapter,
        "sft_adapter_path": sft_adapter,
        "split": split,
        "n_examples": len(examples),
        "probes": [],
    }

    with torch.inference_mode():
        for probe in probes:
            console.print(f"[cyan]Running probe[/cyan] {probe.name} …")
            # Pass nested probe cfg + top-level bias knobs (e.g. position.n)
            nested = dict(bias_cfg.get(probe.name) or {})
            if probe.name == "position" and "n" not in nested and bias_cfg.get("position"):
                nested.update(dict(bias_cfg["position"]))
            result = probe.run(
                tok,
                model,
                examples,
                device=device,
                cfg={**bias_cfg, **nested},
            )
            results.append(result)
            path = out_dir / f"{probe.name}.json"
            write_json(
                path,
                {
                    "name": result.name,
                    "metrics": result.metrics,
                    "details": result.details,
                },
            )
            summary["probes"].append(
                {
                    "name": result.name,
                    "metrics": result.metrics,
                    "path": str(path),
                }
            )

            table = Table(title=f"Probe: {probe.name}")
            table.add_column("Metric")
            table.add_column("Value", justify="right")
            for key in sorted(result.metrics):
                table.add_row(key, str(result.metrics[key]))
            console.print(table)

    write_json(out_dir / "summary.json", summary)
    return summary
