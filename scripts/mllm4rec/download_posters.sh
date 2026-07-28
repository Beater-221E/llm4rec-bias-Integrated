#!/usr/bin/env bash
# TMDb match + poster download. Requires: export TMDB_API_KEY=...
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CFG="${1:-configs/dataset/mllm4rec_ml100k.yaml}"
shift || true
python -m llm4rec_bias_Integrated.data.mllm4rec.cli match-tmdb --config "$CFG" "$@"
python -m llm4rec_bias_Integrated.data.mllm4rec.cli download-posters --config "$CFG" "$@"
