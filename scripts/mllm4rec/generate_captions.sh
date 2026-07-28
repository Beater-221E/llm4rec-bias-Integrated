#!/usr/bin/env bash
# BLIP2 captions (V100: dtype float16 in YAML). Optional: --max-items 20
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CFG="${1:-configs/dataset/mllm4rec_ml100k.yaml}"
shift || true
python -m llm4rec_bias_Integrated.data.mllm4rec.cli generate-captions --config "$CFG" "$@"
