"""Lightweight RuntimeContext — wires hardware/precision/strategy/batch into trainers."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch

from llm4rec.core.exceptions import ConfigurationError
from llm4rec.core.modes import get_mode
from llm4rec.runtime.batch import BatchPlan, resolve_batch_plan
from llm4rec.runtime.hardware import HardwareInfo, apply_nccl_compat_profile, configure_tf32, detect_hardware
from llm4rec.runtime.precision import PrecisionChoice, resolve_precision
from llm4rec.runtime.strategy import (
    StrategyChoice,
    apply_strategy_fallback,
    resolve_strategy,
)


@dataclass
class RuntimeContext:
    """Single runtime object consumed by SFT / GRPO / DPO."""

    cfg: dict[str, Any]
    hardware: HardwareInfo
    precision: PrecisionChoice
    strategy: StrategyChoice
    mode: str
    route: str
    find_unused_parameters: bool = False
    optimization: dict[str, Any] = field(default_factory=dict)
    scaler: torch.cuda.amp.GradScaler | None = None
    model_params_b: float | None = None
    requested_strategy: str = "auto"
    resolved_strategy: str = "single"
    effective_strategy: str = "single"
    fallback_reason: str | None = None
    compile_requested: Any = None
    compile_effective: bool = False
    compile_backend: str | None = None
    compile_mode: str | None = None
    compile_fallback_reason: str | None = None
    has_reference_model: bool = False
    _compiled: dict[int, Any] = field(default_factory=dict, repr=False)
    _log: Any = field(default=print, repr=False)

    # ------------------------------------------------------------------ factories

    @classmethod
    def from_config(cls, cfg: dict[str, Any], *, log=print) -> RuntimeContext:
        apply_nccl_compat_profile()
        hw = detect_hardware()
        mode = get_mode(cfg)
        route = str((cfg.get("experiment") or {}).get("route") or "")
        hw_cfg = cfg.setdefault("hardware", {})
        opt = dict(cfg.get("optimization") or {})

        precision = resolve_precision(hw_cfg.get("precision"), hw, route=route)
        requested = hw_cfg.get("strategy")
        strategy = resolve_strategy(
            requested,
            hw,
            route=route,
            mode=mode,
            model_params_b=None,
            deepspeed=hw_cfg.get("deepspeed"),
            precision=precision.precision,
            stage="sft",
            hw_cfg=hw_cfg,
            free_vram_bytes=hw.free_memory,
        )
        strategy = apply_strategy_fallback(strategy, hw, mode=mode, log=log)

        if precision.precision == "bf16" and hw.tf32_supported:
            configure_tf32(True)

        find_unused = bool(hw_cfg.get("find_unused_parameters", False))

        scaler = None
        if precision.grad_scaler and hw.cuda_available:
            try:
                scaler = torch.amp.GradScaler("cuda")
            except Exception:  # noqa: BLE001
                scaler = torch.cuda.amp.GradScaler()

        compile_cfg = opt.get("compile") or {}
        # Persist resolved settings for preflight / environment dumps
        hw_cfg["precision"] = precision.precision
        hw_cfg["_resolved_precision"] = precision.to_dict()
        hw_cfg["_resolved_strategy"] = strategy.to_dict()
        hw_cfg["_hardware"] = hw.to_dict()
        cfg["optimization"] = opt

        ctx = cls(
            cfg=cfg,
            hardware=hw,
            precision=precision,
            strategy=strategy,
            mode=mode,
            route=route,
            find_unused_parameters=find_unused,
            optimization=opt,
            scaler=scaler,
            requested_strategy=strategy.requested_strategy,
            resolved_strategy=strategy.resolved_strategy or strategy.strategy,
            effective_strategy=strategy.effective_strategy or strategy.strategy,
            fallback_reason=strategy.fallback_reason,
            compile_requested=compile_cfg.get("enabled", "auto"),
            compile_backend=str(compile_cfg.get("backend") or "inductor"),
            compile_mode=str(compile_cfg.get("mode") or "default"),
            _log=log,
        )
        log(
            f"[runtime] precision={precision.precision} amp={precision.amp} "
            f"scaler={precision.grad_scaler} "
            f"strategy requested={ctx.requested_strategy} "
            f"resolved={ctx.resolved_strategy} effective={ctx.effective_strategy} "
            f"world_size={hw.world_size} find_unused={find_unused}"
        )
        return ctx

    def bind_model_params(
        self,
        model: Any,
        *,
        log=print,
        stage: str = "sft",
        has_reference_model: bool | None = None,
    ) -> float:
        """Count params (billions) and optionally re-resolve strategy."""
        n = sum(p.numel() for p in model.parameters())
        self.model_params_b = n / 1e9
        self.strategy.model_params_b = self.model_params_b
        hw_cfg = self.cfg.setdefault("hardware", {})
        stage_l = str(stage or "sft").lower()
        if has_reference_model is None:
            # GRPO/DPO typically hold a reference; SFT does not.
            if stage_l in {"rl", "grpo"}:
                beta = float(
                    (((self.cfg.get("train") or {}).get("rl") or {}).get("grpo") or {}).get(
                        "beta"
                    )
                    or 0.0
                )
                has_reference_model = beta > 0.0
            elif stage_l == "dpo":
                has_reference_model = True
            else:
                has_reference_model = False
        self.has_reference_model = bool(has_reference_model)
        # Re-resolve when strategy was auto and model size now known
        if str(self.requested_strategy).lower() in {"auto", "none", ""}:
            train_block = (self.cfg.get("train") or {}).get(
                "rl" if stage_l in {"rl", "grpo"} else stage_l
            ) or {}
            optimizer = str(train_block.get("optim") or train_block.get("optimizer") or "adamw")
            new = resolve_strategy(
                self.requested_strategy,
                self.hardware,
                route=self.route,
                mode=self.mode,
                model_params_b=self.model_params_b,
                deepspeed=hw_cfg.get("deepspeed"),
                precision=self.precision.precision,
                has_reference_model=bool(has_reference_model),
                optimizer=optimizer,
                stage=stage_l,
                hw_cfg=hw_cfg,
                free_vram_bytes=self.hardware.free_memory,
            )
            new = apply_strategy_fallback(new, self.hardware, mode=self.mode, log=log)
            # Custom GRPO/DPO cannot execute DeepSpeed: prefer FSDP/DDP in auto
            if stage_l in {"rl", "grpo", "dpo"} and new.strategy.startswith("deepspeed"):
                fallback = "fsdp" if (self.model_params_b or 0) >= 3.0 else "ddp"
                log(
                    f"[runtime] custom {stage_l} does not execute DeepSpeed; "
                    f"auto strategy {new.strategy} → {fallback}"
                )
                new.strategy = fallback
                new.effective_strategy = fallback
                new.resolved_strategy = f"{new.resolved_strategy} (auto→{fallback})"
                new.source = "auto_custom_executable"
            if new.effective_strategy != self.effective_strategy:
                log(
                    f"[runtime] strategy re-resolved with model_params_b="
                    f"{self.model_params_b:.3f}: {self.effective_strategy} → {new.effective_strategy}"
                )
            self.strategy = new
            self.resolved_strategy = new.resolved_strategy or new.strategy
            self.effective_strategy = new.effective_strategy or new.strategy
            self.fallback_reason = new.fallback_reason
            hw_cfg["_resolved_strategy"] = new.to_dict()
            if new.memory_estimate:
                hw_cfg["_memory_estimate"] = new.memory_estimate
        hw_cfg["_model_params_b"] = self.model_params_b
        # Never claim ZeRO-2 when effectively on DDP
        if self.effective_strategy == "ddp" and "zero" in str(self.resolved_strategy):
            log(
                f"[runtime] WARNING: resolved={self.resolved_strategy} but "
                f"effective={self.effective_strategy} (do not claim ZeRO)"
            )
        return self.model_params_b

    def resolve_reference_model_strategy(self, *, log=print) -> str:
        """Small policy for the frozen reference model."""
        hw_cfg = self.cfg.get("hardware") or {}
        ref_cfg = hw_cfg.get("reference_model") or {}
        req = str(ref_cfg.get("strategy") or "auto").lower()
        if req in {"replicated", "fsdp", "cpu_offload"}:
            effective = req
        else:
            pb = float(self.model_params_b or 0.0)
            total = int(self.hardware.total_memory or 0)
            # CPU ref + SID-expanded lm_head (T×153k) is host-bound for minutes.
            # Keep 0.5B/1B refs on GPU; only park large refs when VRAM is tight.
            if (
                bool(getattr(self, "has_reference_model", False))
                and total
                and total <= 20 * (1024**3)
                and pb >= 3.0
            ):
                effective = "cpu_offload"
            elif self.world_size <= 1:
                effective = "replicated"
            elif pb >= 7.0:
                effective = "fsdp"
            else:
                effective = "replicated"
        hw_cfg.setdefault("reference_model", {})["_effective_strategy"] = effective
        if self.world_size > 1 and effective != "replicated":
            log(f"[runtime] reference_model strategy={effective} (auto)")
        return effective

    # ------------------------------------------------------------------ properties

    @property
    def device(self) -> torch.device:
        if self.hardware.cuda_available:
            return torch.device(f"cuda:{self.hardware.local_rank}")
        return torch.device("cpu")

    @property
    def world_size(self) -> int:
        return self.hardware.world_size

    @property
    def local_rank(self) -> int:
        return self.hardware.local_rank

    @property
    def dtype(self) -> torch.dtype:
        if self.precision.precision == "bf16":
            return torch.bfloat16
        if self.precision.precision == "fp16":
            return torch.float16
        return torch.float32

    # ------------------------------------------------------------------ AMP / step

    @contextmanager
    def autocast(self) -> Iterator[None]:
        if not self.hardware.cuda_available or self.precision.precision == "fp32":
            with nullcontext():
                yield
            return
        dtype = torch.bfloat16 if self.precision.precision == "bf16" else torch.float16
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            yield

    def backward(self, loss: torch.Tensor) -> None:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        parameters: Any,
        max_grad_norm: float,
    ) -> None:
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            optimizer.step()

    # ------------------------------------------------------------------ model wrap / compile

    def wrap_model(self, model: Any) -> Any:
        """Wrap for distributed training according to effective strategy.

        Custom GRPO/DPO never execute DeepSpeed: fall back to DDP and record reason.
        """
        from llm4rec.core import distributed as dist_utils

        strategy = self.effective_strategy or self.strategy.strategy
        if strategy.startswith("deepspeed"):
            self.resolved_strategy = strategy
            self.strategy.resolved_strategy = strategy
            fallback = "ddp" if self.world_size > 1 else "single"
            reason = "custom_grpo_deepspeed_backend_not_implemented"
            self.log(
                f"[runtime] WARNING: strategy={strategy} not executed by custom "
                f"GRPO/DPO; effective={fallback} ({reason})"
            )
            self.fallback_reason = reason
            self.effective_strategy = fallback
            self.strategy.effective_strategy = fallback
            self.strategy.fallback_reason = reason
            self.strategy.strategy = fallback
            hw_cfg = self.cfg.setdefault("hardware", {})
            hw_cfg["_resolved_strategy"] = self.strategy.to_dict()
            strategy = fallback

        if strategy in {"single", "none"} or self.world_size <= 1:
            return model
        if strategy == "ddp":
            return dist_utils.wrap_ddp(
                model, find_unused_parameters=self.find_unused_parameters
            )
        if strategy == "fsdp":
            # RuntimeContext is the sole precision source of truth.
            pd = None if self.precision.precision == "fp32" else self.dtype
            return dist_utils.wrap_fsdp(
                model,
                param_dtype=pd,
                reduce_dtype=pd,
                buffer_dtype=pd,
            )
        raise ConfigurationError(f"unsupported strategy for wrap_model: {strategy}")

    def maybe_compile(self, model: Any, *, name: str = "model") -> Any:
        """Compile stable tensor regions when optimization.compile.enabled."""
        compile_cfg = self.optimization.get("compile") or {}
        enabled = compile_cfg.get("enabled", "auto")
        self.compile_requested = enabled
        self.compile_backend = str(compile_cfg.get("backend") or "inductor")
        self.compile_mode = str(compile_cfg.get("mode") or "default")
        if enabled in (False, "false", "False", "null", None):
            self.compile_effective = False
            self.compile_fallback_reason = "disabled"
            return model
        if enabled == "auto":
            if self.mode == "reproduction":
                self.compile_effective = False
                self.compile_fallback_reason = "reproduction_auto_off"
                return model
            if not self.hardware.cuda_available:
                self.compile_effective = False
                self.compile_fallback_reason = "no_cuda"
                return model
            cc = self.hardware.compute_capability
            if cc is not None and cc[0] < 8:
                # V100 / pre-Ampere: inductor compile is slow and rarely helps
                self.compile_effective = False
                self.compile_fallback_reason = "pre_ampere_auto_off"
                return model
        if not hasattr(torch, "compile"):
            self.compile_effective = False
            self.compile_fallback_reason = "torch_compile_unavailable"
            return model
        key = id(model)
        if key in self._compiled:
            self.compile_effective = True
            return self._compiled[key]
        backend = self.compile_backend or "inductor"
        mode = self.compile_mode or "default"
        try:
            compiled = torch.compile(model, backend=backend, mode=mode)
            self._compiled[key] = compiled
            self.compile_effective = True
            self.compile_fallback_reason = None
            return compiled
        except Exception as exc:  # noqa: BLE001
            short = str(exc).split("\n")[0][:200]
            self.compile_effective = False
            self.compile_fallback_reason = short
            self.log(f"[compile] Failed; falling back to eager: {short}")
            return model

    # ------------------------------------------------------------------ batch

    def resolve_stage_batch(self, stage: str, block: dict[str, Any]) -> BatchPlan:
        hw_cfg = self.cfg.get("hardware") or {}
        memory_auto = str(hw_cfg.get("memory") or "").lower() == "auto"
        target = block.get("target_global_batch_size", block.get("global_batch_size"))
        plan = resolve_batch_plan(
            world_size=self.world_size,
            per_device_batch_size=int(block.get("per_device_batch_size") or 1),
            gradient_accumulation_steps=block.get("gradient_accumulation_steps"),
            global_batch_size=block.get("global_batch_size"),
            target_global_batch_size=target,
            mode=self.mode,
            memory_auto=memory_auto,
            batch_policy=hw_cfg.get("batch_policy"),
        )
        block["per_device_batch_size"] = plan.per_device_batch_size
        block["gradient_accumulation_steps"] = plan.gradient_accumulation_steps
        block["_effective_global_batch_size"] = plan.effective_global_batch_size
        block["_reference_global_batch"] = plan.reference_global_batch
        block["_relative_batch_deviation"] = plan.relative_batch_deviation
        if plan.message:
            self.log(f"[batch:{stage}] {plan.message}")
        return plan

    def log(self, msg: str) -> None:
        self._log(msg)

    def attention_implementation(self) -> str | None:
        attn = (self.optimization.get("attention") or {}).get("implementation", "auto")
        if attn in (None, "auto"):
            return "sdpa"
        if attn in {"eager", "sdpa", "flash_attention_2"}:
            return str(attn)
        return "sdpa"

    def execution_snapshot(self) -> dict[str, Any]:
        return {
            "precision": self.precision.to_dict(),
            "requested_strategy": self.requested_strategy,
            "resolved_strategy": self.resolved_strategy,
            "effective_strategy": self.effective_strategy,
            "fallback_reason": self.fallback_reason,
            "model_params_b": self.model_params_b,
            "world_size": self.world_size,
            "compile_requested": self.compile_requested,
            "compile_effective": self.compile_effective,
            "compile_backend": self.compile_backend,
            "compile_mode": self.compile_mode,
            "compile_fallback_reason": self.compile_fallback_reason,
            "attention_implementation": self.attention_implementation(),
            "hardware": self.hardware.to_dict(),
        }


def build_runtime(cfg: dict[str, Any], *, log=print) -> RuntimeContext:
    return RuntimeContext.from_config(cfg, log=log)
