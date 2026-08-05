"""wandb 后端。

设计要点：

* **一个 run 贯穿所有 stage**（SFT → eval → RL → eval）。全局 step 连续递增，
  所以在 loss / reward 曲线上能直接看出 RL 从第几步接上，不用去对两个 run 的时间轴。
* 额外记一条 ``progress/stage_id`` 数值序列（sft=0, rl=1, dpo=1, eval=2），
  在 wandb 里叠加显示就是"是否已经进入 RL"的指示线。
* bias 指标是**稀疏**记录的（每 N 步一次），用 ``define_metric`` 声明，
  免得 wandb 把中间的空洞画成断线。
* 只有主进程真正写 wandb；其它 rank 拿到的是一个吞掉所有调用的 no-op。
* wandb 挂掉/没装/没登录**绝不能弄崩训练** —— 所有调用都包了容错，
  真实数据始终有 ``run_dir/metrics.jsonl`` 兜底。
"""

from __future__ import annotations

import os
from typing import Any

# stage → 数值，用于在图上画"现在跑到哪个阶段"
STAGE_IDS: dict[str, int] = {
    "sft": 0,
    "rl": 1,
    "dpo": 1,
    "train_reranker": -1,
    "eval": 2,
}


class NullRun:
    """非主进程 / wandb 关闭时的占位实现，所有调用都是空操作。"""

    enabled = False

    def log(self, *args: Any, **kwargs: Any) -> None:
        return None

    def set_stage(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_summary(self, *args: Any, **kwargs: Any) -> None:
        return None

    def finish(self, *args: Any, **kwargs: Any) -> None:
        return None

    @property
    def url(self) -> str | None:
        return None


class WandbRun:
    """薄封装。全局 step 由我们自己维护，跨 stage 连续。"""

    enabled = True

    def __init__(self, run: Any) -> None:
        self._run = run
        self._step = 0
        self._stage = ""
        self._declare_metrics()

    # ------------------------------------------------------------------ setup
    def _declare_metrics(self) -> None:
        """声明 bias 指标为稀疏序列，避免未采样的 step 被画成断点。"""
        try:
            self._run.define_metric("global_step")
            for prefix in ("train/*", "progress/*"):
                self._run.define_metric(prefix, step_metric="global_step")
            # bias / eval 只在特定 step 有值
            for prefix in ("bias/*", "eval/*"):
                self._run.define_metric(
                    prefix, step_metric="global_step", summary="last"
                )
        except Exception:  # noqa: BLE001 — wandb 的可选特性，失败不影响训练
            pass

    # ------------------------------------------------------------------- 写入
    def set_stage(self, stage: str) -> None:
        """切换阶段。在图上会体现为 ``progress/stage_id`` 的跳变。"""
        self._stage = stage
        self.log({"progress/stage_id": STAGE_IDS.get(stage, -99)}, advance=False)

    def log(
        self,
        metrics: dict[str, Any],
        *,
        step: int | None = None,
        advance: bool = True,
    ) -> None:
        """记一批标量。

        ``step`` 是 stage 内部的局部 step（只用于展示）；真正的横轴是
        我们自己维护的 ``global_step``。
        """
        if not metrics:
            return
        if advance:
            self._step += 1
        payload: dict[str, Any] = {"global_step": self._step}
        if step is not None:
            payload[f"{self._stage}/local_step"] = step
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, (int, float, bool)):
                payload[key] = value
            else:
                payload[key] = str(value)
        try:
            self._run.log(payload)
        except Exception:  # noqa: BLE001
            pass

    def log_summary(self, summary: dict[str, Any]) -> None:
        """写 run summary（最终指标，用于 wandb 表格里横向对比多个 run）。"""
        try:
            for key, value in summary.items():
                self._run.summary[key] = value
        except Exception:  # noqa: BLE001
            pass

    def finish(self) -> None:
        try:
            self._run.finish()
        except Exception:  # noqa: BLE001
            pass

    @property
    def url(self) -> str | None:
        try:
            return str(self._run.url)
        except Exception:  # noqa: BLE001
            return None


def build_wandb_run(
    cfg: dict[str, Any],
    *,
    run_dir: Any,
    experiment_id: str,
    is_main_process: bool = True,
) -> WandbRun | NullRun:
    """按配置初始化 wandb；任何失败都退化成 NullRun，不影响训练。"""
    wb = cfg.get("wandb") or {}
    if not wb.get("enabled", False) or not is_main_process:
        return NullRun()

    mode = str(wb.get("mode") or os.environ.get("WANDB_MODE") or "online")
    if mode == "disabled":
        return NullRun()

    try:
        import wandb
    except ImportError:
        print("[wandb] 未安装 wandb，跳过（指标仍会写 run_dir/metrics.jsonl）")
        return NullRun()

    exp = cfg.get("experiment") or {}
    name = str(exp.get("name") or "run")

    try:
        run = wandb.init(
            project=str(wb.get("project") or os.environ.get("WANDB_PROJECT") or "llm4rec-bias"),
            entity=wb.get("entity") or os.environ.get("WANDB_ENTITY") or None,
            name=os.environ.get("WANDB_NAME") or experiment_id,
            group=str(wb.get("group") or name),
            job_type=wb.get("job_type") or None,
            tags=list(wb.get("tags") or []),
            mode=mode,
            dir=str(run_dir) if run_dir is not None else None,
            config=_flatten_config(cfg),
            reinit=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] 初始化失败（{exc}），退化为本地 jsonl 记录")
        return NullRun()

    return WandbRun(run)


def _flatten_config(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """把嵌套配置摊平成 ``a.b.c`` 形式，方便在 wandb 里按超参分组/过滤。"""
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten_config(value, prefix=f"{full}."))
        elif isinstance(value, (list, tuple)):
            out[full] = list(value)
        else:
            out[full] = value
    return out
