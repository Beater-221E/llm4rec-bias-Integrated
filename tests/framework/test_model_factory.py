"""ModelFactory unit tests (no GPU / HF download required)."""

from __future__ import annotations

import pytest

from llm4rec.components.model.factory import ModelFactory
from llm4rec.components.model.qwen import QWEN_ALIASES, resolve_qwen_checkpoint
from llm4rec.components.model.utils import resolve_precision_name
from llm4rec.core.exceptions import CheckpointError


def test_qwen_aliases_cover_requested_sizes():
    for key in ("qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"):
        assert key in QWEN_ALIASES
        assert resolve_qwen_checkpoint(key).startswith("Qwen/")


def test_resolve_checkpoint_from_config_name():
    ckpt = ModelFactory.resolve_checkpoint({"name": "qwen25_3b", "checkpoint": None})
    assert ckpt == "Qwen/Qwen2.5-3B-Instruct"


def test_resolve_checkpoint_prefers_explicit():
    ckpt = ModelFactory.resolve_checkpoint(
        {"name": "qwen2.5-0.5b", "checkpoint": "local/my-model"}
    )
    assert ckpt == "local/my-model"


def test_resolve_checkpoint_unknown():
    with pytest.raises(CheckpointError):
        ModelFactory.resolve_checkpoint({"name": "not-a-real-model", "checkpoint": None})


def test_precision_aliases():
    assert resolve_precision_name("fp16") == "float16"
    assert resolve_precision_name("bf16") == "bfloat16"
    assert resolve_precision_name("auto") == "auto"
