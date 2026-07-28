#!/usr/bin/env bash
# mllm4rec 全量：dataset → Retriever → Ranker
# 日志：logs/mllm4rec.txt（每次覆盖）；终端只显示单行进度。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
LOG="$ROOT/logs/mllm4rec.txt"
# shellcheck disable=SC1091
source "$ROOT/scripts/run_lib.sh"
activate_bias

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET_CFG="${DATASET_CFG:-mllm4rec_ml100k}"
RETRIEVER_CFG="${RETRIEVER_CFG:-mllm4rec_retriever}"
RANKER_CFG="${RANKER_CFG:-mllm4rec_ranker}"
EXPORT_LRU="${EXPORT_LRU:-experiments/lru/ml-100k}"
EXPORT_RANKER="${EXPORT_RANKER:-experiments/ranker/ml-100k}"
PKL="${PKL:-data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/dataset.pkl}"

log "mllm4rec full start → $LOG"
log "Python=$(command -v python)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
tty_endl "mllm4rec → $LOG"

step env_cuda python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.device_count(), 'GPU(s)', torch.cuda.get_device_name(0))"

NEED_BUILD=1
if [[ -f "$PKL" ]]; then
  if python - "$PKL" >>"$LOG" 2>&1 <<'PY'
import pickle, sys
from pathlib import Path
ds = pickle.load(Path(sys.argv[1]).open("rb"))
ok = all(k in ds for k in ("train", "val", "test", "meta", "umap", "smap"))
caps = len(ds.get("meta_img_des") or {})
print(f"dataset_ok={ok} captions={caps}")
sys.exit(0 if ok and caps > 0 else 1)
PY
  then
    NEED_BUILD=0
    log "OK dataset_reuse $PKL"
    tty_endl "OK dataset_reuse"
  fi
fi

if [[ "$NEED_BUILD" -eq 1 ]]; then
  if [[ -z "${TMDB_API_KEY:-}" ]]; then
    log "ERROR: missing dataset.pkl captions and TMDB_API_KEY unset"
    tty_endl "FAIL mllm4rec — need TMDB_API_KEY or existing captions"
    exit 1
  fi
  step build python -m llm4rec_bias_Integrated.data.mllm4rec.cli build --config "$DATASET_CFG"
fi

step train_retriever python -m llm4rec_bias_Integrated.mllm4rec.cli train-retriever \
  --config "$RETRIEVER_CFG" \
  --export-root "$EXPORT_LRU"

step train_ranker python -m llm4rec_bias_Integrated.mllm4rec.cli train-ranker \
  --config "$RANKER_CFG" \
  --retrieved-pkl "$EXPORT_LRU/retrieved.pkl" \
  --export-root "$EXPORT_RANKER"

log ""
log "Retriever metrics: $EXPORT_LRU/test_metrics.json"
log "Ranker metrics:    $EXPORT_RANKER/subset_metrics.json"
[[ -f "$EXPORT_LRU/test_metrics.json" ]] && log "OK retriever_eval_artifact"
[[ -f "$EXPORT_RANKER/subset_metrics.json" ]] && log "OK ranker_eval_artifact"

log ""
log "mllm4rec FULL DONE"
tty_endl "mllm4rec FULL DONE — see $LOG"
