"""``llm4rec-bias-Integrated evaluate`` — score a run directory or checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table

from llm4rec.core.config import load_config, validate_config
from llm4rec.core.exceptions import ConfigurationError, MissingArtifactError
from llm4rec.core.paths import project_root
from llm4rec.core.reproducibility import write_json
from llm4rec.components.dataset._impl.registry import build_dataset
from llm4rec.components.evaluation._impl.runner import (
    evaluate_checkpoint_on_examples,
    evaluate_run_predictions,
)
from llm4rec.components.model._impl.base import require_cuda
from llm4rec.workflows.grpo4rec import task_spec_from_config

console = Console()


def _override_value(overrides: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for token in overrides:
        if token.startswith(prefix):
            return token.split("=", 1)[1]
    return None


def _resolve_run_dir(cfg: dict[str, Any], overrides: list[str]) -> Path | None:
    # Prefer explicit run_dir=... override / config
    raw = _override_value(overrides, "run_dir")
    if raw is None:
        raw = (cfg.get("evaluation") or {}).get("run_dir")
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else project_root() / path


def _load_run_cfg(run_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Prefer the run's resolved_config so re-eval matches training."""
    resolved = run_dir / "resolved_config.yaml"
    if not resolved.is_file():
        return cfg
    with resolved.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        return cfg
    # Keep CLI evaluation.* overrides on top of the frozen run config.
    merged = dict(loaded)
    if cfg.get("evaluation"):
        merged["evaluation"] = {**(loaded.get("evaluation") or {}), **cfg["evaluation"]}
    return merged


def _is_sid_workflow(cfg: dict[str, Any]) -> bool:
    wf = cfg.get("workflow") or {}
    name = str(wf.get("name") or "").lower()
    task = str(wf.get("task") or "").lower()
    return name == "minionerec" or "semantic_id" in task


def _processed_dir(cfg: dict[str, Any]) -> Path:
    root = Path((cfg.get("paths") or {}).get("data_root", "data"))
    if not root.is_absolute():
        root = project_root() / root
    return root / "processed" / str(cfg["dataset"]["name"])


def _resolve_sid_adapters(
    run_dir: Path,
    overrides: list[str],
    cfg: dict[str, Any],
) -> tuple[str, str | None, str]:
    """Return (adapter_path, sft_adapter_path, tag). Prefer GRPO when present."""
    explicit = _override_value(overrides, "adapter_path")
    sft_override = _override_value(overrides, "sft_adapter_path")
    stage = (
        _override_value(overrides, "checkpoint_stage")
        or (cfg.get("evaluation") or {}).get("checkpoint_stage")
        or ""
    ).strip().lower()

    sft_final = run_dir / "checkpoints" / "sft" / "final"
    grpo_final = run_dir / "checkpoints" / "grpo" / "final"

    if explicit:
        tag = stage or "custom"
        sft = sft_override or (str(sft_final) if sft_final.is_dir() else None)
        return explicit, sft, tag

    if stage == "sft":
        if not sft_final.is_dir():
            raise MissingArtifactError(f"missing SFT adapter at {sft_final}")
        return str(sft_final), None, "sft"

    if stage == "grpo":
        if not grpo_final.is_dir():
            raise MissingArtifactError(f"missing GRPO adapter at {grpo_final}")
        sft = str(sft_final) if sft_final.is_dir() else None
        return str(grpo_final), sft, "grpo"

    # Default: GRPO if available, else SFT
    if grpo_final.is_dir():
        sft = str(sft_final) if sft_final.is_dir() else None
        return str(grpo_final), sft, "grpo"
    if sft_final.is_dir():
        return str(sft_final), None, "sft"

    raise MissingArtifactError(
        f"run_dir={run_dir} has neither checkpoints/sft/final nor checkpoints/grpo/final"
    )


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


def _print_metrics(metrics: dict[str, Any], title: str) -> None:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key in sorted(metrics):
        table.add_row(key, str(metrics[key]))
    console.print(table)


def _run_sid_evaluate(
    cfg: dict[str, Any],
    run_dir: Path,
    overrides: list[str],
) -> int:
    from llm4rec.components.evaluation.generation import evaluate_sid_checkpoint

    require_cuda()
    eval_cfg = cfg.get("evaluation") or {}
    split = str(eval_cfg.get("split") or "test")
    top_k_list = list(eval_cfg.get("top_k") or [1, 5, 10])
    top_k = int(top_k_list[-1])
    max_examples = eval_cfg.get("max_examples")
    free_gen_n = int(eval_cfg.get("free_gen_n", 50))

    adapter, sft_adapter, tag = _resolve_sid_adapters(run_dir, overrides, cfg)
    result = evaluate_sid_checkpoint(
        model_cfg=cfg["model"],
        processed_dir=_processed_dir(cfg),
        adapter_path=adapter,
        sft_adapter_path=sft_adapter,
        split=split,
        top_k=top_k,
        max_examples=max_examples,
        free_gen_n=free_gen_n,
        predictions_dir=run_dir / "predictions",
    )
    out_path = run_dir / "eval" / f"{tag}_metrics.json"
    payload = {
        "tag": tag,
        "adapter_path": adapter,
        "sft_adapter_path": sft_adapter,
        "metrics": result.metrics,
        "slices": result.slices,
        "metadata": result.metadata,
    }
    write_json(out_path, payload)
    _print_metrics(result.metrics, f"SID evaluation ({tag}, {split})")
    console.print(f"[green]Wrote[/green] {out_path}")
    return 0


def run_evaluate(overrides: list[str]) -> int:
    cfg_omega = load_config(overrides)
    cfg = validate_config(cfg_omega)
    run_dir = _resolve_run_dir(cfg, overrides)
    top_k = list((cfg.get("evaluation") or {}).get("top_k") or [1, 5, 10])
    split = str((cfg.get("evaluation") or {}).get("split") or "test")
    use_upstream = bool((cfg.get("evaluation") or {}).get("use_upstream_eval", True))

    if run_dir is None:
        raise ConfigurationError(
            "evaluate requires run_dir=... pointing at a completed training run"
        )
    if not run_dir.is_dir():
        raise ConfigurationError(f"run_dir does not exist: {run_dir}")

    cfg = _load_run_cfg(run_dir, cfg)

    # MiniOneRec / SID: full-catalog beam eval + bias metrics
    if _is_sid_workflow(cfg):
        return _run_sid_evaluate(cfg, run_dir, overrides)

    # Path A: re-aggregate existing predictions (no GPU needed for math)
    pred = run_dir / "predictions" / f"predictions_{split}.jsonl"
    adapter = run_dir / "checkpoints" / "sft" / "final"
    predictions_only = bool(
        (cfg.get("evaluation") or {}).get("predictions_only", False)
    )
    if predictions_only:
        if not pred.is_file():
            raise MissingArtifactError(
                f"evaluation.predictions_only=true but missing {pred}"
            )
        result = evaluate_run_predictions(
            run_dir, split=split, route="letter", top_k=top_k
        )
        _print_metrics(result.metrics, f"Compat metrics ({split})")
        console.print(Pretty(result.metadata.get("metric_sources")))
        return 0

    # Path B: GPU re-eval from adapter if present
    if adapter.is_dir():
        require_cuda()
        dataset = _build_dataset(cfg)
        dataset.preprocess()
        examples = dataset.build_examples(split, task_spec_from_config(cfg))
        result = evaluate_checkpoint_on_examples(
            model_cfg=cfg["model"],
            examples=examples,
            adapter_path=str(adapter),
            top_k=top_k,
            predictions_dir=run_dir / "predictions",
            split=split,
        )
        write_json(
            run_dir / "eval" / f"{split}_metrics.json",
            {
                "metrics": result.metrics,
                "slices": result.slices,
                "metadata": result.metadata,
                "use_upstream_eval": use_upstream,
            },
        )
        _print_metrics(result.metrics, f"Evaluation ({split})")
        console.print("[dim]metric sources[/dim]")
        console.print(Pretty(result.metadata.get("metric_sources")))
        console.print(f"[green]Wrote[/green] {run_dir / 'eval' / f'{split}_metrics.json'}")
        return 0

    if pred.is_file():
        result = evaluate_run_predictions(
            run_dir, split=split, route="letter", top_k=top_k
        )
        _print_metrics(result.metrics, f"Compat metrics ({split})")
        return 0

    raise MissingArtifactError(
        f"run_dir={run_dir} has neither predictions nor checkpoints/sft/final"
    )
