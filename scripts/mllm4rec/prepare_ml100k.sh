#!/usr/bin/env bash
# Prepare official-compatible ml-100k (ml-latest-small) dataset.pkl
# Default: text preprocess only. Pass --with-multimodal to run TMDb+BLIP2 (needs TMDB_API_KEY + GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
WITH_MM=0
ARGS=()
for a in "$@"; do
  if [[ "$a" == "--with-multimodal" ]]; then
    WITH_MM=1
  else
    ARGS+=("$a")
  fi
done
if [[ "$WITH_MM" -eq 1 ]]; then
  python -m llm4rec_bias_Integrated.data.mllm4rec.cli build \
    --config mllm4rec_ml100k \
    "${ARGS[@]}"
else
  python -m llm4rec_bias_Integrated.data.mllm4rec.cli build \
    --config mllm4rec_ml100k \
    --skip-multimodal \
    "${ARGS[@]}"
fi
