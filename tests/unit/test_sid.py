"""Unit tests for SID utilities (no GPU required for parse/quantize math)."""

from __future__ import annotations

import numpy as np
import pytest

from llm4rec_bias_Integrated.semantic_ids.residual_kmeans import break_collisions, residual_quantize
from llm4rec_bias_Integrated.semantic_ids.table import parse_sid, sid_string


def test_sid_string_and_parse() -> None:
    codes = (3, 7, 1, 0)
    text = sid_string(codes)
    assert text == "<s0_3><s1_7><s2_1><s3_0>"
    assert parse_sid(text, 4) == codes
    assert parse_sid("noise <s0_3><s1_7>", 4) is None
    assert parse_sid("<s0_3><s2_1>", 2) is None  # level order break


def test_residual_quantize_unique_after_collision() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8)).astype(np.float64)
    codes = residual_quantize(X, levels=2, K=4, seed=0)
    codes = break_collisions(codes)
    uniq = {tuple(c) for c in codes}
    assert len(uniq) == len(codes)
