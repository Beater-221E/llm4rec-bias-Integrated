"""Stage 编排 —— SFT → eval → RL/DPO/transition/distill → eval 全自动串联。

三条路线共用同一个编排骨架，路线差异只体现在三个 hook：

    build_decoder()   → 怎么把模型输出变成 ranked list
    build_rollout()   → RL 怎么采样（DPO 路线不用）
    build_reward()    → reward 怎么算

每个 stage 结束都存一份完整权重（0.5B 全参 ~2GB），中间不存 ——
bias 的时间序列靠训练中在线评测拿，不靠密集 checkpoint。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm4rec.core import distributed as dist_utils
from llm4rec.core.compose import stage_eval_steps
from llm4rec.core.exceptions import ConfigurationError
from llm4rec.data.base import get_adapter
from llm4rec.data.examples import build_auxiliary_examples, build_examples
from llm4rec.eval.bias import bias_delta, compute_bias_metrics
from llm4rec.eval.catalog import ItemCatalog
from llm4rec.eval.gather import gather_ranked_results
from llm4rec.eval.online import OnlineBiasEvaluator


@dataclass
class StageContext:
    """一次 run 的共享状态。"""

    cfg: dict[str, Any]
    run_dir: Path
    logger: Any
    route: str
    catalog: ItemCatalog
    meta: dict[str, dict[str, Any]]
    train_examples: list[dict[str, Any]]
    val_examples: list[dict[str, Any]]
    test_examples: list[dict[str, Any]]
    sid_table: Any = None
    checkpoint: str | None = None
    summaries: dict[str, Any] = field(default_factory=dict)
    eval_history: list[dict[str, Any]] = field(default_factory=list)
    runtime: Any = None
    artifacts: dict[str, Any] = field(default_factory=dict)


class OnlineEvalHook:
    """挂进 GRPO / DPO 训练循环的在线 bias 评测。

    RL 是按 step 跑的，所以频率也按 step 配（``train.<stage>.bias_eval_steps``，
    留 null 就跟随 ``eval_steps``）。评完直接推 wandb，不落 checkpoint。
    """

    def __init__(
        self,
        evaluator: OnlineBiasEvaluator,
        logger: Any,
        *,
        stage: str,
        every_n_steps: int,
        tokenizer: Any,
    ) -> None:
        self.evaluator = evaluator
        self.logger = logger
        self.stage = stage
        self.every_n_steps = max(1, int(every_n_steps))
        self.tokenizer = tokenizer
        self.history: list[dict[str, Any]] = []

    def on_step(self, step: int, model: Any) -> None:
        if step <= 0 or step % self.every_n_steps:
            return
        self._evaluate(step, model)

    def on_train_end(self, step: int, model: Any) -> None:
        self._evaluate(step, model, final=True)

    def _evaluate(self, step: int, model: Any, *, final: bool = False) -> None:
        try:
            metrics = self.evaluator.evaluate(
                model, self.tokenizer, name=f"{self.stage}_step{step}"
            )
        except Exception as exc:  # noqa: BLE001 — 评测失败不能带崩训练
            self.logger.warning(f"[bias] step {step} 在线评测失败：{exc}")
            return
        metrics["step"] = float(step)
        self.history.append(dict(metrics))
        self.logger.log_metrics(
            {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            stage=self.stage,
            step=step,
            split="bias_online",
            wandb_prefix="bias",
        )
        if final:
            keys = ["pop_lift@1", "delta_gap", "exposure_gini", "tier_gap"]
            brief = " ".join(
                f"{k}={metrics[k]:.4f}" for k in keys if isinstance(metrics.get(k), float)
            )
            self.logger.info(f"[bias] {self.stage} @ step {step}: {brief}")


# ============================================================== 基类


class Pipeline:
    """路线基类。子类实现三个 hook。"""

    route: str = ""

    def __init__(self, cfg: dict[str, Any], run_dir: Path, logger: Any) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.logger = logger
        self.runtime = None

    # ---------------------------------------------------------- 需要子类实现
    def build_decoder(self, ctx: StageContext) -> Any:
        raise NotImplementedError

    def build_rollout(self, ctx: StageContext) -> Any:
        raise NotImplementedError

    def build_reward(self, ctx: StageContext) -> Any:
        raise NotImplementedError

    def prepare_route(self, ctx: StageContext) -> None:
        """路线自己的准备工作（加载 SID 表 / BM25 索引 / reranker）。"""
        return None

    # -------------------------------------------------------------- 通用编排
    def build_context(self) -> StageContext:
        cfg = self.cfg
        # 数据集热插拔：按 data.name 取适配器，代码里不出现任何具体数据集
        adapter = get_adapter(cfg)
        self.adapter = adapter
        processed = adapter.processed_dir(cfg)
        meta = adapter.load_item_meta(cfg)
        catalog = ItemCatalog.from_processed(processed)
        sequences = adapter.user_sequences(adapter.load_interactions(cfg))

        self.logger.info(
            f"[data] {adapter.dataset_key(cfg)}：用户 {len(sequences)}  "
            f"物品 {len(catalog)}  流行度来源 = train split"
        )

        ctx = StageContext(
            cfg=cfg,
            run_dir=self.run_dir,
            logger=self.logger,
            route=self.route,
            catalog=catalog,
            meta=meta,
            train_examples=[],
            val_examples=[],
            test_examples=[],
        )
        self.prepare_route(ctx)

        limits = cfg["data"]
        for split, attr, limit_key in (
            ("train", "train_examples", "max_train_samples"),
            ("val", "val_examples", "max_eval_samples"),
            ("test", "test_examples", "max_eval_samples"),
        ):
            limit = int(limits.get(limit_key) or -1)
            setattr(
                ctx,
                attr,
                build_examples(
                    self.route,
                    sequences=sequences,
                    meta=meta,
                    split=split,
                    cfg=cfg,
                    sid_table=ctx.sid_table,
                    limit=limit if limit > 0 else None,
                ),
            )
        self.logger.info(
            f"[data] 样本 train={len(ctx.train_examples)} "
            f"val={len(ctx.val_examples)} test={len(ctx.test_examples)}"
        )
        return ctx

    def run(self, stages: list[str]) -> dict[str, Any]:
        # torchrun 下初始化进程组。SFT 走 HF Trainer（它自己也会初始化，幂等），
        # RL/DPO 是手写循环，必须靠这里。
        dist_utils.init_process_group()
        from llm4rec.runtime.context import build_runtime

        self.runtime = build_runtime(self.cfg, log=self.logger.info)
        if dist_utils.is_distributed():
            dist_utils.print_distributed_banner(self.logger.info)
            self.logger.info(f"[run] {dist_utils.summary_line()}")
        ctx = self.build_context()
        resume = self.cfg.get("resume_from")
        if resume:
            ctx.checkpoint = str(resume)
            self.logger.info(f"[run] 从 checkpoint 继续：{resume}")

        eval_count = 0
        for stage in stages:
            self.logger.set_stage(stage)
            self.logger.info(f"═══ stage: {stage} ═══")
            if stage == "sft":
                ctx.summaries["sft"] = self.run_sft(ctx)
            elif stage == "rl":
                ctx.summaries["rl"] = self.run_rl(ctx)
            elif stage == "dpo":
                ctx.summaries["dpo"] = self.run_dpo(ctx)
            elif stage == "train_reranker":
                ctx.summaries["train_reranker"] = self.run_train_reranker(ctx)
            elif stage == "transition":
                ctx.summaries["transition"] = self.run_transition(ctx)
            elif stage == "distill":
                ctx.summaries["distill"] = self.run_distill(ctx)
            elif stage == "eval":
                eval_count += 1
                tag = f"eval_{eval_count}"
                ctx.summaries[tag] = self.run_eval(ctx, tag=tag)
            else:
                raise ConfigurationError(f"未知 stage '{stage}'")
            # Ensure all ranks finish the stage (esp. rank0 checkpoint writes)
            # before the next stage loads weights.
            dist_utils.barrier(f"stage_done_{stage}")

        self._write_summary(ctx)
        if dist_utils.is_main():
            from llm4rec.runtime.manifest import build_execution_manifest, write_execution_manifest
            from llm4rec.runtime.profiler import peak_vram_gb

            batch_plans = {}
            for stage, payload in ctx.summaries.items():
                if isinstance(payload, dict) and "batch_plan" in payload:
                    batch_plans[stage] = payload["batch_plan"]
                if isinstance(payload, dict) and isinstance(payload.get("performance"), dict):
                    key = "grpo" if stage == "rl" else stage
                    self.cfg.setdefault("_performance", {}).setdefault(key, {}).update(
                        payload["performance"]
                    )
            manifest = build_execution_manifest(
                self.cfg,
                runtime=self.runtime,
                batch_plans=batch_plans,
                throughput={"peak_vram_gb": peak_vram_gb()},
            )
            path = write_execution_manifest(self.run_dir / "execution_manifest.yaml", manifest)
            self.logger.info(f"[run] execution_manifest → {path}")
        dist_utils.barrier("run_end")
        dist_utils.cleanup()
        return ctx.summaries

    # ------------------------------------------------------------------ SFT
    def run_sft(self, ctx: StageContext) -> dict[str, Any]:
        from llm4rec.core.modes import get_mode
        from llm4rec.trainers.sft import run_sft

        model, tokenizer = self.load_model(ctx)
        sft_cfg = (self.cfg.get("train") or {}).get("sft") or {}
        mode = get_mode(self.cfg)

        # MiniOneRec official mix: SidSFT + SidItemFeat + FusionSeqRec.
        # ctx.train_examples stay seqrec windows so transition/distill keep
        # history + target_item. Only the SFT trainer sees the concatenated mix.
        from llm4rec.data.minionerec_sft import uses_reference_sft

        if uses_reference_sft(self.cfg, route=self.route) and ctx.sid_table is not None:
            from llm4rec.data.minionerec_sft import build_sft_rows, sft_dataset_counts

            objectives = list(
                sft_cfg.get("objectives")
                or ["sid_sft", "sid_item_feat", "fusion_seqrec"]
            )
            # Do NOT collapse by user: keep every sliding window row.
            train_rows = [
                {
                    "user_id": str(ex.get("user_id") or ""),
                    "history": list(ex.get("history") or []),
                    "target_item": ex.get("target_item"),
                }
                for ex in ctx.train_examples
                if ex.get("history") and ex.get("target_item")
            ]
            train = build_sft_rows(
                train_rows=train_rows,
                sid_table=ctx.sid_table,
                meta=ctx.meta,
                objectives=objectives,
            )
            counts = sft_dataset_counts(train)
            self.logger.info(
                f"[sft] MiniOneRec official mix mode={mode} "
                f"objectives={objectives} counts={counts}"
            )
            ctx.summaries.setdefault("sft_dataset_counts", counts)
            eval_rows = [
                {
                    "user_id": str(ex.get("user_id") or ""),
                    "history": list(ex.get("history") or []),
                    "target_item": ex.get("target_item"),
                }
                for ex in ctx.val_examples
                if ex.get("history") and ex.get("target_item")
            ]
            eval_examples = build_sft_rows(
                train_rows=eval_rows,
                sid_table=ctx.sid_table,
                meta=ctx.meta,
                objectives=["sid_sft"],
            )
        else:
            tasks = list(sft_cfg.get("tasks") or [])
            train = list(ctx.train_examples)
            if ctx.sid_table is not None and ({"title2sid", "sid2title"} & set(tasks)):
                aux = build_auxiliary_examples(
                    sid_table=ctx.sid_table, meta=ctx.meta, tasks=tasks
                )
                self.logger.info(f"[sft] 辅助任务样本 {len(aux)}（title2sid / sid2title）")
                train = train + aux
            eval_examples = ctx.val_examples

        summary = run_sft(
            cfg=self.cfg,
            model=model,
            tokenizer=tokenizer,
            train_examples=train,
            eval_examples=eval_examples,
            output_dir=self.run_dir / "sft",
            logger=self.logger,
            runtime=self.runtime,
        )
        ctx.checkpoint = summary["checkpoint"]
        del model
        self._free()
        return summary

    # ------------------------------------------------------------------- RL
    def run_rl(self, ctx: StageContext) -> dict[str, Any]:
        from llm4rec.trainers.grpo import run_grpo

        model, tokenizer = self.load_model(ctx)
        ref_model, _ = self.load_model(ctx, frozen=True)

        rl_examples = ctx.train_examples
        # Official rl.py ConcatDataset: Sid + RLTitle2Sid + RLSeqTitle2Sid.
        # Used for both reproduction and integrated; only SID/SFT differ by mode.
        if self.route == "minionerec" and ctx.sid_table is not None:
            from llm4rec.data.minionerec_rl import (
                build_minionerec_reproduction_rl_train,
                rl_dataset_counts,
            )

            rl_cfg = (self.cfg.get("train") or {}).get("rl") or {}
            rl_rows = [
                {
                    "user_id": str(ex.get("user_id") or ""),
                    "history": list(ex.get("history") or []),
                    "target_item": ex.get("target_item"),
                }
                for ex in ctx.train_examples
                if ex.get("history") and ex.get("target_item")
            ]
            rl_examples = build_minionerec_reproduction_rl_train(
                train_rows=rl_rows,
                sid_table=ctx.sid_table,
                meta=ctx.meta,
                datasets=rl_cfg.get("reference_datasets"),
                seed=int(self.cfg.get("seed") or 42),
            )
            counts = rl_dataset_counts(rl_examples)
            self.logger.info(f"[rl] MiniOneRec official RL mix counts={counts}")
            ctx.summaries.setdefault("rl_dataset_counts", counts)

        hook = self._build_online_hook(ctx, model, tokenizer, stage="rl")
        summary = run_grpo(
            cfg=self.cfg,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            train_examples=rl_examples,
            rollout_fn=self.build_rollout(ctx),
            reward_fn=self.build_reward(ctx),
            output_dir=self.run_dir / "rl",
            logger=self.logger,
            callbacks=[hook] if hook else [],
            stage="rl",
            runtime=self.runtime,
        )
        if hook:
            summary["bias_curve"] = hook.history
        ctx.checkpoint = summary["checkpoint"]
        del model, ref_model
        self._free()
        return summary

    def run_dpo(self, ctx: StageContext) -> dict[str, Any]:
        raise ConfigurationError(f"路线 '{self.route}' 不支持 dpo stage")

    def run_train_reranker(self, ctx: StageContext) -> dict[str, Any]:
        raise ConfigurationError(f"路线 '{self.route}' 不支持 train_reranker stage")

    def run_transition(self, ctx: StageContext) -> dict[str, Any]:
        raise ConfigurationError(f"路线 '{self.route}' 不支持 transition stage")

    def run_distill(self, ctx: StageContext) -> dict[str, Any]:
        raise ConfigurationError(f"路线 '{self.route}' 不支持 distill stage")

    # ----------------------------------------------------------------- eval
    def run_eval(self, ctx: StageContext, *, tag: str) -> dict[str, Any]:
        model, tokenizer = self.load_model(ctx, frozen=True)
        decoder = self.build_decoder(ctx)
        bias_cfg = self.cfg["bias"]
        top_k = int(bias_cfg.get("top_k") or 10)

        limit = bias_cfg.get("final_examples")
        examples = ctx.test_examples
        if limit:
            examples = examples[: int(limit)]

        # 多卡：各 rank 解码一片再汇总。让 rank0 独自跑完全量的话，
        # 其它 rank 会干等在下一个 barrier 上，长时间看起来像卡死。
        shard = dist_utils.shard(examples)
        if dist_utils.is_main():
            self.logger.info(
                f"[{tag}] 评测 {len(examples)} 条（每 rank ≈{len(shard)}），top_k={top_k}"
            )
        local = decoder.decode_batch(
            model,
            tokenizer,
            shard,
            top_k=top_k,
            progress_total=len(examples),
            progress_dir=self.run_dir / "eval" / "progress",
            progress_name=tag,
        )
        results = gather_ranked_results(
            local,
            self.run_dir / "eval" / "shards",
            name=tag,
            timeout_s=float(bias_cfg.get("gather_timeout_s") or 8 * 3600),
        )
        metrics = compute_bias_metrics(
            results,
            ctx.catalog,
            top_k=top_k,
            ips_gamma=float(bias_cfg.get("ips_gamma") or 1.0),
            tier_thresholds=dict(bias_cfg.get("tiers") or {}) or None,
            enabled=list(bias_cfg.get("metrics") or []) or None,
        )

        payload = {"tag": tag, "checkpoint": ctx.checkpoint, "metrics": metrics}
        # 和上一次评测比 —— 这就是"RL 放大了多少 bias"的直接证据
        if ctx.eval_history:
            payload["delta_vs_previous"] = bias_delta(
                ctx.eval_history[-1]["metrics"], metrics
            )
        ctx.eval_history.append(payload)

        if dist_utils.is_main():
            self.logger.log_metrics(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                stage="eval",
                step=len(ctx.eval_history),
                split="test",
                wandb_prefix="eval",
            )
            out = self.run_dir / "eval"
            out.mkdir(parents=True, exist_ok=True)
            text = json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
            (out / f"{tag}.json").write_text(text, encoding="utf-8")
            resume = self.cfg.get("resume_from")
            if resume:
                ckpt = Path(str(resume))
                # eval 单独起 run 时，把指标也挂到 checkpoint 所在 run，方便 notebook 查找
                host = ckpt.parent.parent / "eval"
                if host.parent.is_dir():
                    host.mkdir(parents=True, exist_ok=True)
                    (host / f"{tag}.json").write_text(text, encoding="utf-8")
            self.logger.print_metrics_list(metrics, title=f"{tag} metrics")
        dist_utils.barrier(f"{tag}_done")

        decoder.close()
        del model
        self._free()
        return payload

    # --------------------------------------------------------------- 工具
    def load_model(self, ctx: StageContext, *, frozen: bool = False) -> tuple[Any, Any]:
        raise NotImplementedError

    def _build_online_hook(
        self, ctx: StageContext, model: Any, tokenizer: Any, *, stage: str
    ) -> OnlineEvalHook | None:
        bias_cfg = self.cfg["bias"]
        if stage not in (bias_cfg.get("online_stages") or []):
            self.logger.info(f"[bias] stage '{stage}' 不在 online_stages 里，训练中不评 bias")
            return None

        _, bias_steps = stage_eval_steps(self.cfg, stage)
        stage_cfg = (self.cfg["train"].get(stage) or {})
        n_examples = int(
            stage_cfg.get("eval_examples") or bias_cfg.get("online_examples") or 256
        )
        pool = ctx.val_examples or ctx.test_examples
        evaluator = OnlineBiasEvaluator.from_config(
            self.cfg,
            decoder=self.build_decoder(ctx),
            catalog=ctx.catalog,
            pool=pool,
            n_examples=n_examples,
            seed=int(self.cfg.get("seed") or 42),
            shard_dir=self.run_dir / "eval" / "online_shards",
        )
        self.logger.info(
            f"[bias] 在线评测已启用：每 {bias_steps} step 评一次，"
            f"固定子集 {len(evaluator.examples)} 条（不落 checkpoint）"
        )
        return OnlineEvalHook(
            evaluator,
            self.logger,
            stage=stage,
            every_n_steps=bias_steps,
            tokenizer=tokenizer,
        )

    def _write_summary(self, ctx: StageContext) -> None:
        if not dist_utils.is_main():
            return
        payload = {
            "experiment": self.cfg["experiment"],
            "stages": ctx.summaries,
            "eval_history": ctx.eval_history,
            "final_checkpoint": ctx.checkpoint,
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        if len(ctx.eval_history) >= 2:
            first, last = ctx.eval_history[0]["metrics"], ctx.eval_history[-1]["metrics"]
            delta = bias_delta(first, last)
            (self.run_dir / "eval" / "bias_delta.json").write_text(
                json.dumps(delta, indent=2, default=str) + "\n", encoding="utf-8"
            )
            self.logger.print_metrics_list(
                delta, title="bias 变化（首次评测 → 末次评测）"
            )
            self.logger.wandb.log_summary(delta)

    @staticmethod
    def _free() -> None:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ====================================================== 路线 1：MiniOneRec


class MiniOneRecPipeline(Pipeline):
    route = "minionerec"

    def prepare_route(self, ctx: StageContext) -> None:
        from llm4rec.sid.build import resolve_for_training
        from llm4rec.sid.table import SidTable

        sid_dir = resolve_for_training(self.cfg)
        ctx.sid_table = SidTable(sid_dir)
        self.logger.info(
            f"[sid] 加载静态产物 {sid_dir}\n"
            f"      物品 {len(ctx.sid_table)}  层数 {ctx.sid_table.levels}  "
            f"碰撞率 {ctx.sid_table.manifest.collision_rate:.4f}"
        )

    def load_model(self, ctx: StageContext, *, frozen: bool = False) -> tuple[Any, Any]:
        from llm4rec.sid.model import load_for_sid, load_trained, resolve_checkpoint

        runtime = self.runtime or ctx.runtime
        attn = runtime.attention_implementation() if runtime is not None else None
        lr = dist_utils.local_rank()
        from llm4rec.runtime.precision import weight_dtype_name

        if runtime is not None:
            dtype_name = weight_dtype_name(runtime.precision, trainable=not frozen)
        else:
            dtype_name = str(self.cfg["hardware"].get("precision") or "fp32")
        if ctx.checkpoint:
            bundle = load_trained(
                ctx.checkpoint,
                dtype=dtype_name,
                local_rank=lr,
                attn_implementation=attn,
                backbone=resolve_checkpoint(self.cfg["model"]),
                sid_table=ctx.sid_table,
            )
        else:
            model_cfg = dict(self.cfg["model"])
            model_cfg["dtype"] = dtype_name
            model_cfg["_experiment_mode"] = str(self.cfg.get("mode") or "integrated")
            bundle = load_for_sid(
                model_cfg,
                ctx.sid_table,
                local_rank=lr,
                attn_implementation=attn,
                experiment_mode=model_cfg["_experiment_mode"],
            )
        # Strategy binding is trainer/stage-owned (see run_sft / run_grpo / run_dpo).
        if runtime is not None:
            ctx.runtime = runtime
        if frozen:
            bundle.model.eval()
            for p in bundle.model.parameters():
                p.requires_grad = False
        return bundle.model, bundle.tokenizer

    def build_decoder(self, ctx: StageContext) -> Any:
        from llm4rec.decoders.constrained_beam import ConstrainedBeamDecoder

        dec = self.cfg["decoder"]
        return ConstrainedBeamDecoder(
            ctx.sid_table,
            num_beams=int(dec.get("num_beams") or 20),
            max_new_tokens=dec.get("max_new_tokens"),
            fail_on_invalid=bool(dec.get("fail_on_invalid", True)),
        )

    def build_rollout(self, ctx: StageContext) -> Any:
        from llm4rec.runtime.kv_cache import resolve_kv_cache
        from llm4rec.trainers.rollouts import ConstrainedBeamRollout

        # Constrained beam + prefix_allowed_tokens_fn → dynamic cache only.
        resolve_kv_cache(self.cfg, constrained=True)
        rl = (self.cfg.get("train") or {}).get("rl") or {}
        grpo = rl.get("grpo") or {}
        return ConstrainedBeamRollout(
            ctx.sid_table,
            do_sample=bool(grpo.get("do_sample", True)),
            temperature=float(grpo.get("temperature") or 1.0),
            length_penalty=float(grpo.get("length_penalty") or 0.0),
            beam_search=bool(grpo.get("beam_search", True)),
        )

    def build_reward(self, ctx: StageContext) -> Any:
        from llm4rec.trainers.rewards import make_minionerec_reward

        # Official mix has reference_target_text, not always target_item.
        # Default to upstream rule + ndcg_rule (rl.py reward_type=ranking).
        rl_cfg = dict(self.cfg["train"]["rl"])
        reward_cfg = dict(rl_cfg.get("reward") or {})
        reward_cfg.setdefault("implementation", "minionerec_reference")
        rl_cfg["reward"] = reward_cfg
        return make_minionerec_reward(ctx.sid_table, rl_cfg)

    def run_transition(self, ctx: StageContext) -> dict[str, Any]:
        from llm4rec.sid.transition import run_transition

        summary = run_transition(
            cfg=self.cfg,
            sid_table=ctx.sid_table,
            catalog=ctx.catalog,
            train_examples=ctx.train_examples,
            val_examples=ctx.val_examples,
            output_dir=self.run_dir / "transition",
            logger=self.logger,
        )
        # Teacher artifact only — do not replace the SFT / LLM checkpoint.
        ctx.artifacts["transition_checkpoint"] = summary["checkpoint"]
        self.logger.info(
            f"[transition] teacher → {summary['checkpoint']} "
            f"(ctx.checkpoint 仍为 {ctx.checkpoint})"
        )
        return summary

    def run_distill(self, ctx: StageContext) -> dict[str, Any]:
        from llm4rec.trainers.sid_distill import run_sid_distill

        model, tokenizer = self.load_model(ctx)
        hook = self._build_online_hook(ctx, model, tokenizer, stage="distill")
        summary = run_sid_distill(
            cfg=self.cfg,
            model=model,
            tokenizer=tokenizer,
            sid_table=ctx.sid_table,
            catalog=ctx.catalog,
            train_examples=ctx.train_examples,
            eval_examples=ctx.val_examples,
            output_dir=self.run_dir / "distill",
            logger=self.logger,
            runtime=self.runtime,
            artifacts=ctx.artifacts,
            callbacks=[hook] if hook else [],
        )
        if hook:
            summary["bias_curve"] = hook.history
        ctx.checkpoint = summary["checkpoint"]
        del model
        self._free()
        return summary


# ========================================================== 路线 2：Rec-R1


class RecR1Pipeline(Pipeline):
    route = "recr1"

    def prepare_route(self, ctx: StageContext) -> None:
        from llm4rec.retrieval.bm25 import BM25Index, index_path_for

        self.retriever = BM25Index.load(index_path_for(self.cfg))
        self.logger.info(
            f"[bm25] 加载索引：{len(self.retriever.item_ids)} 物品，"
            f"{len(self.retriever.term_docs)} 词项"
        )

    def load_model(self, ctx: StageContext, *, frozen: bool = False) -> tuple[Any, Any]:
        from llm4rec.sid.model import load_trained, resolve_checkpoint, resolve_dtype
        from transformers import AutoModelForCausalLM, AutoTokenizer

        runtime = self.runtime or ctx.runtime
        attn = runtime.attention_implementation() if runtime is not None else "sdpa"
        lr = dist_utils.local_rank()
        from llm4rec.runtime.precision import weight_dtype_name

        if runtime is not None:
            dtype_name = weight_dtype_name(runtime.precision, trainable=not frozen)
        else:
            dtype_name = str(self.cfg["hardware"].get("precision") or "fp32")
        if ctx.checkpoint:
            bundle = load_trained(
                ctx.checkpoint,
                dtype=dtype_name,
                local_rank=lr,
                attn_implementation=attn,
                backbone=resolve_checkpoint(self.cfg["model"]),
            )
            model, tokenizer = bundle.model, bundle.tokenizer
        else:
            # Rec-R1 不需要 SID 词表扩展，直接加载 backbone
            checkpoint = resolve_checkpoint(self.cfg["model"])
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            load_kwargs: dict[str, Any] = {
                "dtype": resolve_dtype(dtype_name),
                "attn_implementation": attn or "sdpa",
            }
            try:
                model = AutoModelForCausalLM.from_pretrained(checkpoint, **load_kwargs)
            except Exception:
                load_kwargs["attn_implementation"] = "eager"
                model = AutoModelForCausalLM.from_pretrained(checkpoint, **load_kwargs)
            import torch

            if torch.cuda.is_available():
                model = model.to(f"cuda:{lr}")
        # Strategy binding is trainer/stage-owned.
        if runtime is not None:
            ctx.runtime = runtime
        if frozen:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
        return model, tokenizer

    def build_decoder(self, ctx: StageContext) -> Any:
        from llm4rec.decoders.bm25_query import BM25QueryDecoder

        dec = self.cfg["decoder"]
        retr = dec.get("retriever") or {}
        return BM25QueryDecoder(
            self.retriever,
            answer_key=str(dec.get("answer_key") or "query"),
            max_new_tokens=int(dec.get("max_new_tokens") or 512),
            retrieval_top_k=int(retr.get("eval_top_k") or 100),
        )

    def build_rollout(self, ctx: StageContext) -> Any:
        from llm4rec.runtime.kv_cache import generation_cache_kwargs, resolve_kv_cache
        from llm4rec.trainers.rollouts import SamplingRollout

        grpo = self.cfg["train"]["rl"]["grpo"]
        choice = resolve_kv_cache(self.cfg, constrained=False)
        cache_kwargs = generation_cache_kwargs(choice)
        return SamplingRollout(
            temperature=float(grpo.get("temperature") or 0.6),
            top_p=float(grpo.get("top_p") or 0.95),
            max_new_tokens=int(self.cfg["train"]["rl"].get("max_response_length") or 512),
            use_cache=bool(cache_kwargs.get("use_cache", True)),
            cache_implementation=cache_kwargs.get("cache_implementation"),
            kv_choice=choice,
            cfg=self.cfg,
        )

    def build_reward(self, ctx: StageContext) -> Any:
        from llm4rec.trainers.rewards import make_recr1_reward

        return make_recr1_reward(self.retriever, self.cfg)


# ========================================================= 路线 3：DPO4Rec


class DPO4RecPipeline(Pipeline):
    route = "dpo4rec"

    def prepare_route(self, ctx: StageContext) -> None:
        from llm4rec.rerankers.service import RerankerService

        self.service = RerankerService(self.cfg, sorted(ctx.meta.keys()))
        self.popularity = dict(ctx.catalog.counts)
        self.logger.info(
            f"[reranker] {self.service.rr_cfg.get('kind', 'prm')} "
            f"候选列表长度 {self.service.candidate_size}"
        )

    def build_context(self) -> StageContext:
        ctx = super().build_context()
        # 给每条样本预生成固定的候选列表（固定 seed → 跨 step/跨 run 可比）
        rng = random.Random(int(self.cfg.get("seed") or 42))
        queued = [*ctx.train_examples, *ctx.val_examples, *ctx.test_examples]
        self.service.assign_candidates(
            queued,
            self.popularity,
            rng,
            desc="reranker/candidates",
            logger=self.logger,
        )
        self.logger.info(
            "[reranker] 候选列表已固定（KAR：最多 "
            f"{self.service.n_positives} 个后续正例 + 未交互物品均匀负例，固定 seed）"
        )
        return ctx

    def load_model(self, ctx: StageContext, *, frozen: bool = False) -> tuple[Any, Any]:
        return RecR1Pipeline.load_model(self, ctx, frozen=frozen)  # type: ignore[arg-type]

    def build_decoder(self, ctx: StageContext) -> Any:
        from llm4rec.decoders.knowledge_reranker import KnowledgeRerankerDecoder

        return KnowledgeRerankerDecoder(
            self.service,
            max_new_tokens=int((self.cfg["train"].get("dpo") or {}).get("max_new_tokens") or 512),
        )

    def build_rollout(self, ctx: StageContext) -> Any:
        raise ConfigurationError("DPO4Rec 路线不用 GRPO rollout")

    def build_reward(self, ctx: StageContext) -> Any:
        from llm4rec.trainers.rewards import make_dpo4rec_scorer

        return make_dpo4rec_scorer(self.service, self.cfg)

    def run_train_reranker(self, ctx: StageContext) -> dict[str, Any]:
        summary = self.service.train(
            ctx.train_examples,
            self.popularity,
            logger=self.logger,
            seed=int(self.cfg.get("seed") or 42),
        )
        out = self.run_dir / "reranker"
        self.service.save(out)
        self.service.release_cuda()
        summary["path"] = str(out)
        self.logger.info(f"[reranker] 训练完成 → {out}")
        return summary

    def run_dpo(self, ctx: StageContext) -> dict[str, Any]:
        from llm4rec.trainers.dpo import run_dpo

        self.service.ensure_device()
        model, tokenizer = self.load_model(ctx)
        ref_model, _ = self.load_model(ctx, frozen=True)
        hook = self._build_online_hook(ctx, model, tokenizer, stage="dpo")
        score_fn = self.build_reward(ctx)

        def on_iteration_end(iteration: int, reasoning: dict[str, str]) -> None:
            # 论文 §IV-C-2 的双向增益：用当前最好的推理文本再训一轮 reranker
            self.logger.info(f"[dpo] 迭代 {iteration} 结束，用最优推理文本回灌 reranker")
            self.service.train(
                ctx.train_examples,
                self.popularity,
                logger=self.logger,
                seed=int(self.cfg.get("seed") or 42),
                reasoning_by_user=reasoning,
            )

        summary = run_dpo(
            cfg=self.cfg,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            train_examples=ctx.train_examples,
            score_fn=score_fn,
            output_dir=self.run_dir / "dpo",
            logger=self.logger,
            callbacks=[hook] if hook else [],
            on_iteration_end=on_iteration_end,
            runtime=self.runtime,
        )
        if hook:
            summary["bias_curve"] = hook.history
        summary.pop("best_reasoning", None)  # 太大，不进 summary.json
        ctx.checkpoint = summary["checkpoint"]
        del model, ref_model
        self._free()
        return summary


PIPELINES = {
    "minionerec": MiniOneRecPipeline,
    "recr1": RecR1Pipeline,
    "dpo4rec": DPO4RecPipeline,
}


def build_pipeline(cfg: dict[str, Any], run_dir: Path, logger: Any) -> Pipeline:
    route = str(cfg["experiment"]["route"])
    if route not in PIPELINES:
        raise ConfigurationError(f"未知路线 '{route}'，可用：{sorted(PIPELINES)}")
    return PIPELINES[route](cfg, run_dir, logger)
