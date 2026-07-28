"""``llm4rec-bias-Integrated train`` — SFT / GRPO / evaluate (Phases 3–5)."""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from llm4rec_bias_Integrated.core.config import load_config, validate_config
from llm4rec_bias_Integrated.core.context import build_run_dir, create_context
from llm4rec_bias_Integrated.core.exceptions import ConfigurationError
from llm4rec_bias_Integrated.core.paths import project_root
from llm4rec_bias_Integrated.core.reproducibility import write_json
from llm4rec_bias_Integrated.datasets.registry import build_dataset
from llm4rec_bias_Integrated.evaluation.hacking import analyze_reward_hacking, write_hacking_report
from llm4rec_bias_Integrated.models.base import require_cuda
from llm4rec_bias_Integrated.models.loader import load_model_bundle
from llm4rec_bias_Integrated.trainers.distributed import (
    allocate_shared_run_dir,
    distributed_barrier,
    resolve_distributed_plan,
)
from llm4rec_bias_Integrated.workflows.grpo4rec import task_spec_from_config
from llm4rec_bias_Integrated.workflows.registry import build_workflow

console = Console()


def _resolve_data_root(cfg: dict[str, Any]) -> Path:
    root = Path((cfg.get("paths") or {}).get("data_root", "data"))
    return root if root.is_absolute() else project_root() / root


def _build_dataset(cfg: dict[str, Any]):
    ds_cfg = cfg["dataset"]
    return build_dataset(
        str(ds_cfg["name"]),
        data_root=_resolve_data_root(cfg),
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
        train_ratio=float(ds_cfg.get("train_ratio", 0.8)),
        val_ratio=float(ds_cfg.get("val_ratio", 0.1)),
    )


def _stages_to_run(cfg: dict[str, Any]) -> list[str]:
    stages = list((cfg.get("training") or {}).get("stages") or ["sft"])
    allowed = {"sft", "grpo", "evaluate", "analyze"}
    filtered = [s for s in stages if s in allowed]
    if not filtered:
        filtered = ["sft"]
    return filtered


def _maybe_multigpu_relaunch(overrides: list[str], cfg: dict[str, Any]) -> int | None:
    if os.environ.get("LLM4REC_FULL_DISTRIBUTED_CHILD") == "1":
        return None
    training = cfg.get("training") or {}
    if training.get("auto_launch_multi_gpu", True) is False:
        return None
    plan = resolve_distributed_plan(training, model_name=str(cfg["model"].get("name")))
    if plan.strategy == "single" or plan.nproc_per_node <= 1:
        return None
    if plan.details.get("already_launched"):
        return None

    console.print(
        f"[cyan]Detected {plan.nproc_per_node} GPUs[/cyan] — "
        f"re-launching with Accelerate (strategy={plan.strategy})."
    )
    from llm4rec_bias_Integrated.tracking.inplace_progress import write_progress_status

    write_progress_status(
        f"launching multi-GPU ({plan.nproc_per_node} processes)…"
    )
    # Warm HF cache in the single parent process so N ranks do not race Hub/SSL.
    ckpt = str((cfg.get("model") or {}).get("checkpoint") or "")
    if ckpt:
        try:
            from transformers import AutoConfig, AutoTokenizer

            console.print(f"[dim]Warming HF cache for {ckpt}…[/dim]")
            write_progress_status(f"warming HF cache: {ckpt}…")
            AutoTokenizer.from_pretrained(ckpt)
            AutoConfig.from_pretrained(ckpt)
        except Exception as exc:  # noqa: BLE001 — best-effort cache warm
            console.print(f"[yellow]HF cache warm skipped: {exc}[/yellow]")
    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(plan.nproc_per_node),
        "--multi_gpu",
        "-m",
        "llm4rec_bias_Integrated.cli.main",
        "train",
        *overrides,
    ]
    env = os.environ.copy()
    env["LLM4REC_FULL_DISTRIBUTED_CHILD"] = "1"
    # Isolate file-rendezvous dirs across sequential multi-GPU launches that
    # often reuse the same MASTER_PORT (default 29500). Stale barrier marks
    # from a prior job otherwise let some ranks pass early while others hang.
    env["LLM4REC_FULL_JOB_ID"] = uuid.uuid4().hex
    # Prefer hardware.env from YAML; fall back to safe NCCL defaults for V100/P2P.
    hw_env = ((cfg.get("hardware") or {}).get("env") or {}) if isinstance(cfg, dict) else {}
    for key, value in hw_env.items():
        if value is not None:
            env.setdefault(str(key), str(value))
    env.setdefault("NCCL_NVML_ENABLE", "0")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    # Avoid 4-way Hub metadata races ("httpx client has been closed" / SSL EOF).
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    console.print(f"[dim]{' '.join(cmd)}[/dim]")
    # Keep progress TTY fd open across accelerate relaunch (Python defaults
    # close_fds=True which would drop fd 3).
    pass_fds: tuple[int, ...] = ()
    fd_raw = os.environ.get("LLM4REC_PROGRESS_FD", "").strip()
    if fd_raw.isdigit():
        fd = int(fd_raw)
        if fd > 2:
            try:
                os.fstat(fd)
                pass_fds = (fd,)
            except OSError:
                pass
    return subprocess.call(
        cmd, env=env, cwd=str(project_root()), pass_fds=pass_fds
    )


def _resolve_init_checkpoint(cfg: dict[str, Any], overrides: list[str]) -> str | None:
    for token in overrides:
        if token.startswith("init_checkpoint="):
            return token.split("=", 1)[1]
    return cfg.get("init_checkpoint") or (cfg.get("training") or {}).get("init_checkpoint")


def _run_heldout_eval(
    *,
    cfg: dict[str, Any],
    workflow,
    dataset,
    task_spec,
    adapter_path: str | None,
    sft_adapter_path: str | None,
    plan,
    context,
    tag: str,
) -> dict[str, Any]:
    # GRPO adapters are trained on merged-SFT weights → merge SFT first.
    tok, model, _ = load_model_bundle(
        cfg["model"],
        peft_cfg=None,
        adapter_path=adapter_path,
        sft_adapter_path=sft_adapter_path,
        for_training=False,
        local_rank=plan.local_rank,
    )
    workflow._tok, workflow._model = tok, model
    evaluator = workflow.build_evaluator(context)
    test_examples = workflow.build_examples_with_spec(dataset, "test", task_spec)
    result = evaluator.evaluate(model, test_examples, split=f"test_{tag}")
    payload = {
        "tag": tag,
        "adapter_path": adapter_path,
        "metrics": result.metrics,
        "slices": result.slices,
        "metadata": result.metadata,
    }
    write_json(context.run_dir / "eval" / f"{tag}_metrics.json", payload)
    return payload


def run_train(overrides: list[str]) -> int:
    # Resolve hardware profile (CUDA_VISIBLE_DEVICES / NCCL_*) before CUDA init.
    cfg_omega = load_config(overrides)
    require_cuda()
    cfg = validate_config(cfg_omega)

    # Allow init_checkpoint override into config
    init_override = _resolve_init_checkpoint(cfg, overrides)
    if init_override:
        cfg["init_checkpoint"] = init_override

    relaunch_code = _maybe_multigpu_relaunch(overrides, cfg)
    if relaunch_code is not None:
        return int(relaunch_code)

    from llm4rec_bias_Integrated.tracking.inplace_progress import (
        install_inplace_progress,
        write_progress_status,
    )

    # Re-bind progress TTY after accelerate spawn (pts path / env may settle late).
    install_inplace_progress(force=True)
    write_progress_status("train: preparing run…")

    plan = resolve_distributed_plan(
        cfg.get("training") or {}, model_name=str(cfg["model"].get("name"))
    )
    # Multi-GPU: all ranks must share one run_dir (otherwise SFT adapter is only on rank0's path).
    shared_run_dir = allocate_shared_run_dir(cfg, plan, build_run_dir=build_run_dir)
    distributed_barrier(plan, name="run_dir")
    context = create_context(
        cfg_omega,
        cli_overrides=overrides,
        create_run_dir=True,
        run_dir=shared_run_dir,
    )
    # Re-apply init into context config (create_context validated from omega)
    if init_override:
        context.config["init_checkpoint"] = init_override
    context.logger.experiment_id = context.experiment_id

    if plan.is_main_process:
        context.logger.info(f"Run directory: {context.run_dir}")
        write_json(
            context.run_dir / "distributed_plan.json",
            {
                "strategy": plan.strategy,
                "world_size": plan.world_size,
                "nproc_per_node": plan.nproc_per_node,
                "effective_batch_size": plan.effective_batch_size,
                "launch_hint": plan.launch_hint,
                "details": plan.details,
            },
        )

    dataset = _build_dataset(cfg)
    dataset.download()
    dataset.preprocess()
    task_spec = task_spec_from_config(cfg)

    workflow_name = str(cfg["workflow"]["name"])
    is_sid = workflow_name == "minionerec"
    workflow = build_workflow(workflow_name)

    if is_sid:
        # Ensure SID artifacts exist (prepare may have built them)
        from llm4rec_bias_Integrated.semantic_ids.build import (
            build_semantic_ids,
            build_sid_dataset,
            sid_dir,
        )

        processed = Path((cfg.get("paths") or {}).get("data_root", "data"))
        if not processed.is_absolute():
            processed = project_root() / processed
        processed = processed / "processed" / str(cfg["dataset"]["name"])
        out = sid_dir(processed)
        if not (out / "semantic_ids.json").is_file() or not (out / "sid_train.jsonl").is_file():
            sid_cfg = (cfg.get("workflow") or {}).get("semantic_id") or {}
            table_path = build_semantic_ids(
                processed_dir=processed,
                levels=int(sid_cfg.get("levels", 3)),
                codebook_size=int(sid_cfg.get("codebook_size", 64)),
                seed=int(cfg["experiment"]["seed"]),
            )
            build_sid_dataset(
                processed_dir=processed,
                sid_table_path=table_path,
                history_max_length=int(cfg["dataset"].get("history_max_length", 8)),
                train_per_user=int(cfg["dataset"].get("train_per_user", 4)),
                seed=int(cfg["experiment"]["seed"]),
                train_limit=cfg["dataset"].get("train_limit"),
                eval_limit=cfg["dataset"].get("eval_limit"),
            )
        workflow.set_examples([], [])
    else:
        train_examples = workflow.build_examples_with_spec(dataset, "train", task_spec)
        eval_examples = workflow.build_examples_with_spec(dataset, "validation", task_spec)
        if not train_examples:
            raise ConfigurationError("No train examples produced")
        workflow.set_examples(train_examples, eval_examples)

    stages = _stages_to_run(cfg)
    summaries: dict[str, Any] = {"stages": stages, "workflow": workflow_name}
    hacking_points: list[dict[str, Any]] = []

    sft_adapter = None
    grpo_adapter = None

    def _sid_eval(adapter: str | None, sft: str | None, tag: str) -> dict[str, Any]:
        payload = workflow.evaluate_sid(
            context,
            adapter_path=adapter,
            sft_adapter_path=sft,
            split="test",
            tag=tag,
        )
        write_json(context.run_dir / "eval" / f"{tag}_metrics.json", payload)
        return payload

    multi_gpu = plan.world_size > 1

    def _record_eval(tag: str, payload: dict[str, Any], *, train_reward: float = 0.0) -> None:
        summaries[tag] = payload
        metrics = payload.get("metrics") or {}

        def _metric_float(key: str, *fallback: str) -> float:
            for k in (key, *fallback):
                if k not in metrics:
                    continue
                value = metrics[k]
                if value is None or value == "not_applicable":
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
            return 0.0

        point = {
            "checkpoint": "sft" if tag == "eval_sft" else "grpo",
            "train/reward": float(train_reward),
            "eval/hr@10": _metric_float("hr@10", "hr@1"),
            "eval/hr@1": _metric_float("hr@1"),
            "eval/ndcg@10": _metric_float("ndcg@10"),
        }
        if tag == "eval_grpo":
            point["train/kl"] = (summaries.get("grpo") or {}).get("last_kl")
        hacking_points.append(point)

    if "sft" in stages:
        trainer = workflow.build_trainer(context)
        summaries["sft"] = trainer.train(context)
        sft_adapter = summaries["sft"].get("adapter_path")
        del trainer
        distributed_barrier(plan, name="after_sft")
        # Multi-GPU: defer held-out eval until after GRPO so other ranks do not sit in
        # an NCCL barrier while rank0 reloads a second model (looks like util=100%/low W).
        if plan.is_main_process and not multi_gpu:
            if is_sid:
                _record_eval("eval_sft", _sid_eval(sft_adapter, None, "sft"))
            else:
                _record_eval(
                    "eval_sft",
                    _run_heldout_eval(
                        cfg=cfg,
                        workflow=workflow,
                        dataset=dataset,
                        task_spec=task_spec,
                        adapter_path=sft_adapter,
                        sft_adapter_path=None,
                        plan=plan,
                        context=context,
                        tag="sft",
                    ),
                )
        elif plan.is_main_process and multi_gpu:
            context.logger.info(
                "Multi-GPU: deferring post-SFT held-out eval until training finishes"
            )
        distributed_barrier(plan, name="after_sft_eval", prefer_file=True)

    if "grpo" in stages:
        init_for_grpo = sft_adapter or context.config.get("init_checkpoint")
        if not init_for_grpo and not (cfg.get("training") or {}).get("allow_grpo_from_base"):
            raise ConfigurationError(
                "GRPO requires an SFT adapter or init_checkpoint=... "
                "(or set training.allow_grpo_from_base=true)"
            )
        if not hasattr(workflow, "build_grpo_trainer"):
            raise ConfigurationError(f"workflow '{workflow_name}' has no GRPO trainer")
        grpo_trainer = workflow.build_grpo_trainer(
            context, sft_adapter_path=init_for_grpo
        )
        summaries["grpo"] = grpo_trainer.train(context)
        grpo_adapter = summaries["grpo"].get("adapter_path")
        del grpo_trainer
        distributed_barrier(plan, name="after_grpo")
        if plan.is_main_process:
            # Single-GPU: only grpo eval here (sft already done). Multi-GPU: both.
            # Resume path: stages may omit sft but still pass init_checkpoint — eval it too.
            sft_for_eval = sft_adapter or (
                init_for_grpo if "sft" not in stages else None
            )
            if multi_gpu and sft_for_eval:
                if is_sid:
                    _record_eval("eval_sft", _sid_eval(sft_for_eval, None, "sft"))
                else:
                    _record_eval(
                        "eval_sft",
                        _run_heldout_eval(
                            cfg=cfg,
                            workflow=workflow,
                            dataset=dataset,
                            task_spec=task_spec,
                            adapter_path=sft_for_eval,
                            sft_adapter_path=None,
                            plan=plan,
                            context=context,
                            tag="sft",
                        ),
                    )
            if is_sid:
                _record_eval(
                    "eval_grpo",
                    _sid_eval(grpo_adapter, init_for_grpo, "grpo"),
                    train_reward=float(summaries["grpo"].get("last_reward") or 0.0),
                )
            else:
                _record_eval(
                    "eval_grpo",
                    _run_heldout_eval(
                        cfg=cfg,
                        workflow=workflow,
                        dataset=dataset,
                        task_spec=task_spec,
                        adapter_path=grpo_adapter,
                        sft_adapter_path=init_for_grpo,
                        plan=plan,
                        context=context,
                        tag="grpo",
                    ),
                    train_reward=float(summaries["grpo"].get("last_reward") or 0.0),
                )
        distributed_barrier(plan, name="after_grpo_eval", prefer_file=True)

    if "evaluate" in stages and "grpo" not in stages and "sft" in stages:
        summaries.setdefault("evaluate", summaries.get("eval_sft"))

    if "evaluate" in stages and grpo_adapter:
        summaries["evaluate"] = summaries.get("eval_grpo")

    if "analyze" in stages and len(hacking_points) >= 2 and plan.is_main_process:
        report = analyze_reward_hacking(hacking_points)
        write_hacking_report(context.run_dir / "eval" / "reward_hacking.json", report)
        summaries["reward_hacking"] = report
        gap = (report.get("gaps") or {}).get("relative") or {}
        context.logger.info(
            f"Reward hacking gap (relative)={gap.get('hacking_gap')} "
            f"Δreward={gap.get('delta_reward_raw')} ΔHR@10={gap.get('delta_heldout_raw')}"
        )

    if plan.is_main_process:
        write_json(context.run_dir / "summary.json", summaries)
        console.print("[green]Train finished[/green]")
        overview = Table(title="Run overview", show_header=True)
        overview.add_column("Field", style="cyan")
        overview.add_column("Value")
        overview.add_row("run_dir", str(context.run_dir))
        overview.add_row("sft_adapter", str((summaries.get("sft") or {}).get("adapter_path")))
        overview.add_row("grpo_adapter", str((summaries.get("grpo") or {}).get("adapter_path")))
        console.print(overview)

        for tag in ("eval_sft", "eval_grpo"):
            metrics = (summaries.get(tag) or {}).get("metrics") or {}
            if not metrics:
                continue
            table = Table(title=tag, show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")
            for key, value in metrics.items():
                if isinstance(value, float):
                    rendered = f"{value:.6g}"
                else:
                    rendered = str(value)
                table.add_row(str(key), rendered)
            console.print(table)

        hacking = summaries.get("reward_hacking") or {}
        gaps = ((hacking.get("gaps") or {}).get("relative")) or {}
        if gaps:
            hack_table = Table(title="reward_hacking (relative)", show_header=True)
            hack_table.add_column("Metric", style="cyan")
            hack_table.add_column("Value", justify="right")
            for key in (
                "hacking_gap",
                "delta_reward_raw",
                "delta_heldout_raw",
                "delta_reward_norm",
                "delta_heldout_norm",
            ):
                if key in gaps:
                    value = gaps[key]
                    rendered = f"{value:.6g}" if isinstance(value, float) else str(value)
                    hack_table.add_row(key, rendered)
            console.print(hack_table)
    # File barrier: other ranks must not exit while main writes summary / analyzes.
    distributed_barrier(plan, name="train_done", prefer_file=True)
    return 0
