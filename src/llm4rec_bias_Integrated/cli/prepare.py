"""``llm4rec-bias-Integrated prepare`` — download, preprocess, print acceptance stats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table

from llm4rec_bias_Integrated.core.config import load_config, validate_config
from llm4rec_bias_Integrated.core.paths import project_root
from llm4rec_bias_Integrated.core.schemas import TaskSpec
from llm4rec_bias_Integrated.datasets.registry import build_dataset

console = Console()


def _task_spec_from_config(cfg: dict[str, Any]) -> TaskSpec:
    ds = cfg["dataset"]
    wf = cfg.get("workflow") or {}
    task = str(wf.get("task") or "candidate_choice")
    return TaskSpec(
        task=task,
        history_max_length=int(ds.get("history_max_length", 20)),
        candidate_size=int(ds.get("candidate_size", 10)),
        negative_sampling=str(ds.get("negative_sampling", "uniform")),
        target_position=str(ds.get("target_position", "random")),
        framing=str(ds.get("framing", "neutral")),
    )


def _resolve_data_root(cfg: dict[str, Any]) -> Path:
    paths = cfg.get("paths") or {}
    root = paths.get("data_root", "data")
    path = Path(root)
    if not path.is_absolute():
        path = project_root() / path
    return path


def prepare_from_config(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    data = validate_config(cfg)

    ds_cfg = data["dataset"]
    adapter = build_dataset(
        str(ds_cfg["name"]),
        data_root=_resolve_data_root(data),
        rating_threshold=float(ds_cfg.get("rating_threshold", 4.0)),
        split=str(ds_cfg.get("split", "leave_one_out")),
        history_max_length=int(ds_cfg.get("history_max_length", 20)),
        candidate_size=int(ds_cfg.get("candidate_size", 10)),
        negative_sampling=str(ds_cfg.get("negative_sampling", "uniform")),
        target_position=str(ds_cfg.get("target_position", "random")),
        framing=str(ds_cfg.get("framing", "neutral")),
        min_user_interactions=int(ds_cfg.get("min_user_interactions", 5)),
        seed=int(data["experiment"]["seed"]),
        train_limit=ds_cfg.get("train_limit"),
        eval_limit=ds_cfg.get("eval_limit"),
        train_ratio=float(ds_cfg.get("train_ratio", 0.8)),
        val_ratio=float(ds_cfg.get("val_ratio", 0.1)),
    )
    adapter.download()
    adapter.preprocess()
    summary = adapter.summary()
    task_spec = _task_spec_from_config(data)
    workflow_name = str((data.get("workflow") or {}).get("name") or "")

    sid_payload: dict[str, Any] | None = None
    if workflow_name == "minionerec" or bool((data.get("workflow") or {}).get("build_sid")):
        from llm4rec_bias_Integrated.semantic_ids.build import build_semantic_ids, build_sid_dataset

        wf = data.get("workflow") or {}
        sid_cfg = wf.get("semantic_id") or {}
        table_path = build_semantic_ids(
            processed_dir=adapter.processed_dir,
            levels=int(sid_cfg.get("levels", 3)),
            codebook_size=int(sid_cfg.get("codebook_size", 64)),
            seed=int(data["experiment"]["seed"]),
        )
        paths = build_sid_dataset(
            processed_dir=adapter.processed_dir,
            sid_table_path=table_path,
            history_max_length=int(ds_cfg.get("history_max_length", 8)),
            train_per_user=int(ds_cfg.get("train_per_user", 4)),
            seed=int(data["experiment"]["seed"]),
            with_titles=not bool(ds_cfg.get("sid_only", False)),
            train_limit=ds_cfg.get("train_limit"),
            eval_limit=ds_cfg.get("eval_limit"),
        )
        sid_payload = {
            "semantic_ids": str(table_path),
            "sid_train": str(paths["train"]),
            "sid_val": str(paths["val"]),
            "sid_test": str(paths["test"]),
        }
        console.print("[cyan]Built SID artifacts[/cyan]")
        console.print(Pretty(sid_payload))

    train_examples = []
    first_example = None
    if workflow_name != "minionerec":
        train_examples = adapter.build_examples("train", task_spec)
        first_example = train_examples[0] if train_examples else None
    payload = {
        **summary,
        "n_train_examples": len(train_examples),
        "sid": sid_payload,
        "first_train_example": None
        if first_example is None
        else {
            "example_id": first_example.example_id,
            "user_id": first_example.user_id,
            "history_item_ids": first_example.history_item_ids,
            "target_item_id": first_example.target_item_id,
            "candidates": first_example.candidates,
            "target_text": first_example.target_text,
            "target_index": first_example.target_index,
            "features_keys": sorted(first_example.features.keys()),
            "prompt_preview": first_example.prompt_messages[-1]["content"][:400],
        },
    }
    # Persist acceptance dump next to processed data
    out = adapter.processed_dir / "prepare_summary.json"
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return payload


def print_prepare_report(payload: dict[str, Any]) -> None:
    console.print("[bold green]Dataset prepare complete[/bold green]")
    table = Table(title="Dataset summary", show_header=False)
    table.add_column("key")
    table.add_column("value")
    table.add_row("dataset", str(payload.get("name")))
    table.add_row("n_interactions (shape)", str(payload.get("n_interactions")))
    table.add_row("n_users", str(payload.get("n_users")))
    table.add_row("n_items_catalog", str(payload.get("n_items_catalog")))
    table.add_row("n_items_with_train_pop", str(payload.get("n_items_with_train_pop")))
    table.add_row("fingerprint", str(payload.get("fingerprint")))
    table.add_row("split_method", str(payload.get("split_method")))
    table.add_row("rating_threshold", str(payload.get("rating_threshold")))
    table.add_row("seed", str(payload.get("seed")))
    console.print(table)

    splits = payload.get("split_sizes") or {}
    split_table = Table(title="Split sizes")
    split_table.add_column("split")
    split_table.add_column("interactions", justify="right")
    split_table.add_column("users", justify="right")
    split_table.add_column("items", justify="right")
    for name, n_key, u_key, i_key in (
        ("train", "n_train", "n_users_train", "n_items_train"),
        ("validation", "n_validation", "n_users_validation", "n_items_validation"),
        ("test", "n_test", "n_users_test", "n_items_test"),
    ):
        split_table.add_row(
            name,
            str(splits.get(n_key)),
            str(splits.get(u_key)),
            str(splits.get(i_key)),
        )
    console.print(split_table)

    console.print("[bold]First normalized interaction[/bold]")
    console.print(Pretty(payload.get("first_interaction")))
    console.print("[bold]Popularity statistics[/bold]")
    console.print(Pretty(payload.get("popularity")))
    console.print(
        f"[bold]Train examples[/bold]: {payload.get('n_train_examples')}"
    )
    console.print("[bold]First train example[/bold]")
    console.print(Pretty(payload.get("first_train_example")))


def run_prepare(overrides: list[str]) -> int:
    cfg = load_config(overrides)
    data = validate_config(cfg)
    # Ensure experiment seed is present even for prepare-only
    payload = prepare_from_config(data)
    print_prepare_report(payload)
    return 0
