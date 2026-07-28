#!/usr/bin/env bash
# grpo4rec 全量：prepare → SFT → GRPO → evaluate
# 日志：logs/grpo4rec.txt（每次覆盖）；终端只显示单行进度。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
LOG="$ROOT/logs/grpo4rec.txt"
# shellcheck disable=SC1091
source "$ROOT/scripts/run_lib.sh"
activate_bias

HARDWARE="${HARDWARE:-multi}"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export LLM4REC_COMPOSE="hardware=${HARDWARE} scale=full"
if [[ "$HARDWARE" == "multi" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export NCCL_NVML_ENABLE="${NCCL_NVML_ENABLE:-0}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

log "grpo4rec full start → $LOG"
log "Python=$(command -v python)  COMPOSE=$LLM4REC_COMPOSE  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
tty_endl "grpo4rec → $LOG"

step validate python -m llm4rec_bias_Integrated.cli.main validate experiment=smoke_grpo
step prepare  python -m llm4rec_bias_Integrated.cli.main prepare experiment=smoke_test dataset=movielens_100k
step train    python -m llm4rec_bias_Integrated.cli.main train experiment=smoke_grpo

RUN_DIR="$(ls -dt runs/movielens_100k/grpo4rec/*/seed_*/20* 2>/dev/null | head -1 || true)"
if [[ -n "${RUN_DIR:-}" && -d "$RUN_DIR" ]]; then
  step evaluate python -m llm4rec_bias_Integrated.cli.main evaluate "run_dir=${RUN_DIR}"
  log "run_dir=$RUN_DIR"
else
  log "WARN: no run_dir found for post-eval"
fi

log ""
log "grpo4rec FULL DONE"
tty_endl "grpo4rec FULL DONE — see $LOG"
