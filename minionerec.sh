#!/usr/bin/env bash
# minionerec 全量：SID prepare → SFT → GRPO → evaluate
# 日志：logs/minionerec.txt（每次覆盖）；终端只显示单行进度。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
LOG="$ROOT/logs/minionerec.txt"
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

log "minionerec full start → $LOG"
log "Python=$(command -v python)  COMPOSE=$LLM4REC_COMPOSE  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
tty_endl "minionerec → $LOG"

step validate python -m llm4rec.cli.main validate experiment=smoke_sid
step prepare  python -m llm4rec.cli.main prepare experiment=smoke_sid
step train    python -m llm4rec.cli.main train experiment=smoke_sid

RUN_DIR="$(ls -dt runs/movielens_100k/minionerec/*/seed_*/20* 2>/dev/null | head -1 || true)"
if [[ -n "${RUN_DIR:-}" && -d "$RUN_DIR" ]]; then
  step evaluate python -m llm4rec.cli.main evaluate "run_dir=${RUN_DIR}"
  log "run_dir=$RUN_DIR"
else
  log "WARN: no run_dir found for post-eval"
fi

log ""
log "minionerec FULL DONE"
tty_endl "minionerec FULL DONE — see $LOG"
