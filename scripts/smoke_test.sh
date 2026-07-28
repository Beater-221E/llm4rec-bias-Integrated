#!/usr/bin/env bash
# Phase 1 smoke: config composition + CLI help.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== llm4rec-bias-Integrated --help =="
llm4rec-bias-Integrated --help

echo
echo "== validate experiment=smoke_test =="
llm4rec-bias-Integrated validate experiment=smoke_test

echo
echo "== validate model alias qwen2.5-1b =="
llm4rec-bias-Integrated validate experiment=smoke_test model=qwen2.5-1b

echo
echo "== prepare movielens_100k (uses cache if present) =="
llm4rec-bias-Integrated prepare experiment=smoke_test dataset=movielens_100k | head -n 40

echo
echo "Phase 1+2 smoke OK"
