#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python -m llm4rec_bias_Integrated.data.mllm4rec.cli build \
  --config mllm4rec_ml1m \
  "$@"
