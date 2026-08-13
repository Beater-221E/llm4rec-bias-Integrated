"""Centralized hardware / precision / distributed runtime."""

from __future__ import annotations

from llm4rec.runtime.batch import BatchPlan, resolve_batch_plan
from llm4rec.runtime.context import RuntimeContext, build_runtime
from llm4rec.runtime.hardware import HardwareInfo, detect_hardware
from llm4rec.runtime.precision import PrecisionChoice, resolve_precision
from llm4rec.runtime.strategy import StrategyChoice, resolve_strategy
from llm4rec.runtime.preflight import run_preflight, format_preflight_table

__all__ = [
    "HardwareInfo",
    "detect_hardware",
    "PrecisionChoice",
    "resolve_precision",
    "StrategyChoice",
    "resolve_strategy",
    "BatchPlan",
    "resolve_batch_plan",
    "RuntimeContext",
    "build_runtime",
    "run_preflight",
    "format_preflight_table",
]
