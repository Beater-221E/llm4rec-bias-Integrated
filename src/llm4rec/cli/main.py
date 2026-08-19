"""llm4rec-bias-Integrated 统一 CLI。

    python -m llm4rec.cli.main validate       --config minionerec_qwen05b_amazon
    python -m llm4rec.cli.main download-data  --config minionerec_qwen05b_amazon
    python -m llm4rec.cli.main prepare-data   --config minionerec_qwen05b_amazon
    python -m llm4rec.cli.main embed-items   --config minionerec_qwen05b_amazon
    python -m llm4rec.cli.main build-sid     --config minionerec_qwen05b_amazon
    python -m llm4rec.cli.main build-bm25    --config recr1_qwen05b_amazon
    python -m llm4rec.cli.main run           --config minionerec_qwen05b_amazon --stages sft,eval,rl,eval

一般不用直接敲这些 —— ``bash prepare.sh`` 和 ``bash run.sh`` 已经包好了。
任意 ``a.b.c=value`` 可以跟在后面覆盖配置。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from llm4rec.core.compose import compose, to_dict, validate
from llm4rec.core.exceptions import ConfigurationError, LabError, MissingArtifactError
from llm4rec.core.paths import project_root
from llm4rec.core.reproducibility import set_seed

COMMANDS = (
    "list",
    "validate",
    "download-data",
    "prepare-data",
    "embed-items",
    "build-sid",
    "build-bm25",
    "run",
)


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(prog="llm4rec", description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument(
        "--config", default=None, help="configs/exp/<name>.yaml 的名字（list 不需要）"
    )
    parser.add_argument("--stages", default=None, help="逗号分隔，覆盖配置里的 stages")
    parser.add_argument("--resume-from", default=None, help="从某个 checkpoint 目录继续")
    parser.add_argument("--force", action="store_true", help="强制重建已有产物")
    return parser.parse_known_args(argv)


def _load(args: argparse.Namespace, overrides: list[str]) -> dict[str, Any]:
    cfg = to_dict(compose(args.config, overrides))
    if args.stages:
        cfg["stages"] = [s.strip() for s in args.stages.split(",") if s.strip()]
    if args.resume_from:
        cfg["resume_from"] = args.resume_from
    return validate(cfg)


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def _shared_run_stamp(parent: Path) -> str:
    """Same timestamp on every rank.

    ``datetime.now()`` per process can roll over a second boundary
    (this job created both ``..._101526`` and ``..._101527``). Online
    eval then waits forever for ``.done`` files in the other directory.
    """
    env = str(os.environ.get("LLM4REC_RUN_TS") or "").strip()
    if env:
        return env
    port = str(os.environ.get("MASTER_PORT") or "single")
    marker = parent / f".run_stamp_{port}"
    if _is_main_process():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(stamp + "\n", encoding="utf-8")
        return stamp
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if marker.is_file():
            text = marker.read_text(encoding="utf-8").strip()
            if text:
                return text
        time.sleep(0.05)
    raise ConfigurationError(
        f"rank {os.environ.get('RANK', '?')} timed out waiting for shared run stamp {marker}"
    )


def _build_run_dir(cfg: dict[str, Any]) -> Path:
    exp, data = cfg["experiment"], cfg["data"]
    from llm4rec.data.base import get_adapter

    root = Path(cfg["paths"].get("runs_dir") or "runs")
    if not root.is_absolute():
        root = project_root() / root
    parent = (
        root
        / get_adapter(cfg).dataset_key(cfg)
        / str(exp["route"])
        / str(cfg["model"]["name"]).replace("/", "_")
        / f"seed_{cfg['seed']}"
    )
    path = parent / _shared_run_stamp(parent)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _print_plan(cfg: dict[str, Any]) -> None:
    exp, model, data = cfg["experiment"], cfg["model"], cfg["data"]
    train = cfg.get("train") or {}
    lines = [
        f"实验          : {exp['name']}  (route={exp['route']})",
        f"阶段          : {' → '.join(cfg['stages'])}",
        f"数据          : {data['name']} / {data['category']}",
        f"Backbone      : {model['checkpoint']}  精度={model['dtype']}  "
        f"全参={model.get('full_finetuning')}",
        f"解码器        : {cfg['decoder']['name']}",
        f"随机种子      : {cfg['seed']}",
    ]
    for stage in ("sft", "rl", "dpo"):
        block = train.get(stage)
        if not block:
            continue
        bits = [f"lr={block.get('learning_rate')}"]
        bits.append(
            f"max_steps={block['max_steps']}"
            if block.get("max_steps")
            else f"epochs={block.get('epochs')}"
        )
        if stage in ("rl", "dpo"):
            bits.append(f"eval_steps={block.get('eval_steps')}")
            bits.append(
                f"bias_eval_steps={block.get('bias_eval_steps') or block.get('eval_steps')}"
            )
        lines.append(f"  {stage:12s}: {'  '.join(str(b) for b in bits)}")
    online = cfg["bias"].get("online_stages") or []
    lines.append(f"bias 在线评测 : {online or '关闭'}（SFT 阶段按设计不评）")
    wb = cfg["wandb"]
    lines.append(f"wandb         : {wb.get('mode')} / {wb.get('project')}")
    print("\n".join(lines))


# --------------------------------------------------------------------- 命令


def cmd_list() -> int:
    """列出所有可跑的实验 + 数据集 + backbone —— 跑实验前先看这个。"""
    from llm4rec.core.paths import configs_dir
    from llm4rec.data.base import available_datasets

    root = configs_dir()
    rows = []
    for path in sorted((root / "exp").glob("*.yaml")):
        try:
            cfg = validate(to_dict(compose(path.stem)))
        except Exception as exc:  # noqa: BLE001
            rows.append((path.stem, "配置有误", str(exc)[:40], "", ""))
            continue
        route = cfg["experiment"]["route"]
        key = "dpo" if route == "dpo4rec" else "rl"
        block = (cfg.get("train") or {}).get(key) or {}
        rows.append(
            (
                path.stem,
                route,
                str(cfg["model"]["checkpoint"]).split("/")[-1],
                f"{cfg['data']['name']}/{cfg['data'].get('category') or cfg['data'].get('variant')}",
                ",".join(cfg["stages"]),
            )
        )

    widths = [max(len(str(r[i])) for r in rows) for i in range(5)]
    header = ("实验名", "route", "backbone", "数据集", "stages")
    widths = [max(widths[i], len(header[i])) for i in range(5)]
    print("可跑的实验（EXP=<实验名> bash run.sh）：\n")
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print("─" * (sum(widths) + 8))
    for row in rows:
        print("  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))

    print(f"\n可用数据集 (data.name)  : {available_datasets()}")
    print(f"可用 backbone          : {sorted(p.stem for p in (root / 'model').glob('*.yaml'))}")
    print(f"可用 DeepSpeed 预设    : {sorted(p.stem for p in (root / 'deepspeed').glob('*.yaml'))}")
    print(
        "\n常用覆盖：\n"
        "  换卡        GPUS=0,1,2,3 bash run.sh\n"
        "  换数据集    bash run.sh data.name=movielens data.variant=ml-1m\n"
        "  换 backbone bash run.sh model.checkpoint=Qwen/Qwen2.5-3B-Instruct\n"
        "  开 DeepSpeed DEEPSPEED=zero2 bash run.sh\n"
        "  改评测频率  bash run.sh train.rl.eval_steps=10 train.rl.bias_eval_steps=50\n"
        "  导入现成SID bash run.sh sid.import_from=/data/shared/sid/amazon23_ind\n"
        "  只评测      STAGES=eval RESUME_FROM=runs/.../rl/final bash run.sh"
    )
    return 0


def cmd_validate(cfg: dict[str, Any]) -> int:
    from llm4rec.runtime.preflight import run_preflight

    _print_plan(cfg)
    print(f"mode           : {cfg.get('mode') or (cfg.get('experiment') or {}).get('mode')}")
    run_preflight(cfg)
    print("\n配置校验通过 ✓")
    return 0


def cmd_download_data(cfg: dict[str, Any], force: bool) -> int:
    from llm4rec.data.base import get_adapter

    adapter = get_adapter(cfg)
    print(f"[download] 数据集 {adapter.dataset_key(cfg)}")
    for label, (url, dest) in adapter.raw_files(cfg).items():
        print(f"[download]   {label}: {url}")
    adapter.download(cfg, force=force)
    missing = adapter.check_raw(cfg)
    if missing:
        raise MissingArtifactError("下载后仍缺文件：\n  " + "\n  ".join(missing))
    print("[download] 原始文件齐全 ✓")
    return 0


def cmd_prepare_data(cfg: dict[str, Any], force: bool) -> int:
    from llm4rec.data.base import get_adapter

    adapter = get_adapter(cfg)
    missing = adapter.check_raw(cfg)
    if missing:
        raise MissingArtifactError(
            "缺少原始文件：\n  " + "\n  ".join(missing)
            + "\n跑 `STEPS=download bash prepare.sh` 自动下载。"
        )
    adapter.preprocess(cfg, force=force)
    return 0


def cmd_embed_items(cfg: dict[str, Any], force: bool) -> int:
    from llm4rec.data.base import get_adapter
    from llm4rec.sid.embeddings import encode_items

    if "sid" not in cfg:
        print(f"[embed] 路线 '{cfg['experiment']['route']}' 不需要 SID embedding，跳过")
        return 0
    meta = get_adapter(cfg).load_item_meta(cfg)
    item_ids = sorted(meta.keys())
    texts = [str(meta[i].get("text") or meta[i].get("title") or "") for i in item_ids]
    encode_items(cfg, item_ids, texts, force=force)
    return 0


def cmd_build_sid(cfg: dict[str, Any], force: bool) -> int:
    from llm4rec.sid.build import build_sid

    if "sid" not in cfg:
        print(f"[sid] 路线 '{cfg['experiment']['route']}' 不用 SID，跳过")
        return 0
    build_sid(cfg, force=force)
    return 0


def cmd_build_bm25(cfg: dict[str, Any], force: bool) -> int:
    from llm4rec.retrieval.bm25 import build_and_save

    if str((cfg.get("decoder") or {}).get("name")) != "bm25_query":
        print(f"[bm25] 路线 '{cfg['experiment']['route']}' 不用 BM25，跳过")
        return 0
    build_and_save(cfg, force=force)
    return 0


def cmd_run(cfg: dict[str, Any]) -> int:
    from llm4rec.core.reproducibility import collect_environment, write_json
    from llm4rec.pipeline import build_pipeline
    from llm4rec.runtime.hardware import apply_nccl_compat_profile
    from llm4rec.runtime.preflight import run_preflight
    from llm4rec.tracking.logger import build_logger

    set_seed(int(cfg["seed"]))
    main_process = _is_main_process()
    apply_nccl_compat_profile()
    run_dir = _build_run_dir(cfg)

    exp = cfg["experiment"]
    experiment_id = (
        f"{exp['name']}_{cfg['model']['name']}_seed{cfg['seed']}_"
        f"{datetime.now().strftime('%m%d_%H%M')}"
    )
    logger = build_logger(
        cfg.get("tracking") or {},
        run_dir,
        full_config=cfg,
        experiment_id=experiment_id,
        is_main_process=main_process,
    )

    if main_process:
        _print_plan(cfg)
        print(f"mode           : {cfg.get('mode')}")
        print(f"\n输出目录      : {run_dir}")
        if logger.wandb.url:
            print(f"wandb run     : {logger.wandb.url}")
        run_preflight(cfg, log=logger.info if hasattr(logger, "info") else print)
        write_json(run_dir / "resolved_config.json", cfg)
        env = collect_environment(project_root())
        env["experiment_id"] = experiment_id
        env["hardware"] = (cfg.get("hardware") or {}).get("_hardware")
        write_json(run_dir / "environment.json", env)
    else:
        # Non-main ranks still need batch/precision resolution for identical training.
        run_preflight(cfg, log=lambda *_a, **_k: None)

    pipeline = build_pipeline(cfg, run_dir, logger)
    try:
        summaries = pipeline.run(list(cfg["stages"]))
    finally:
        logger.finish()

    if main_process:
        print(f"\n完成。summary → {run_dir / 'summary.json'}")
        for stage, payload in summaries.items():
            if isinstance(payload, dict) and payload.get("checkpoint"):
                print(f"  {stage:12s} → {payload['checkpoint']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args, overrides = _parse_args(list(argv if argv is not None else sys.argv[1:]))
    try:
        if args.command == "list":
            return cmd_list()
        if not args.config:
            raise ConfigurationError("除 list 外都必须指定 --config")
        cfg = _load(args, overrides)
        handlers = {
            "validate": lambda: cmd_validate(cfg),
            "download-data": lambda: cmd_download_data(cfg, args.force),
            "prepare-data": lambda: cmd_prepare_data(cfg, args.force),
            "embed-items": lambda: cmd_embed_items(cfg, args.force),
            "build-sid": lambda: cmd_build_sid(cfg, args.force),
            "build-bm25": lambda: cmd_build_bm25(cfg, args.force),
            "run": lambda: cmd_run(cfg),
        }
        return handlers[args.command]()
    except ConfigurationError as exc:
        print(f"\n[配置错误] {exc}\n", file=sys.stderr)
        return 2
    except LabError as exc:
        print(f"\n[{type(exc).__name__}] {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
