#!/usr/bin/env bash
# minionerec SID 独立评估（默认评 SFT）
#
# 用法：
#   1. 在下方 RUN_DIRS / CHECKPOINTS 里填路径（可只填 RUN_DIRS）
#   2. ./evaluation.sh
#
# 环境变量（可选）：
#   CHECKPOINT_STAGE=sft|grpo   默认 sft
#   EXPERIMENT=smoke_sid|...    run 无 resolved_config 时的 fallback
#   CUDA_VISIBLE_DEVICES=0
#   HARDWARE=single|multi
#
# smoke_sid：config 里的冒烟实验名（少量数据 + 几步训练，用来通流程），
# 不是正式全量评测。正式评测请填真实 run_dir。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
LOG="$ROOT/logs/evaluation.txt"
# shellcheck disable=SC1091
source "$ROOT/scripts/run_lib.sh"
activate_bias

# =============================================================================
# 填这里：要评的 run / checkpoint（默认留空，避免误跑）
# =============================================================================

# 完整训练 run 目录（含 checkpoints/ 与 resolved_config.yaml）
# 例: "runs/movielens_100k/minionerec/<exp>/seed_42/<timestamp>"
RUN_DIRS=(
)

# 可选：显式 adapter 路径，须与 RUN_DIRS 一一对应；留空则用
#   $run_dir/checkpoints/${CHECKPOINT_STAGE}/final
# 例: "runs/.../checkpoints/sft/final"
CHECKPOINTS=(
)

# =============================================================================

CHECKPOINT_STAGE="${CHECKPOINT_STAGE:-sft}"
EXPERIMENT="${EXPERIMENT:-smoke_sid}"
HARDWARE="${HARDWARE:-single}"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export LLM4REC_COMPOSE="hardware=${HARDWARE}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log "evaluation start → $LOG"
log "Python=$(command -v python)  stage=${CHECKPOINT_STAGE}  experiment=${EXPERIMENT}"
log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
tty_endl "evaluation → $LOG"

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  log "ERROR: RUN_DIRS 为空。请编辑 evaluation.sh 填入 run 目录后再跑。"
  log ""
  log "示例："
  log '  RUN_DIRS=('
  log '    "runs/movielens_100k/minionerec/<exp>/seed_42/<timestamp>"'
  log '  )'
  log ""
  log "可选同时指定 adapter（与 RUN_DIRS 等长）："
  log '  CHECKPOINTS=('
  log '    "runs/.../checkpoints/sft/final"'
  log '  )'
  tty_endl "evaluation FAILED — fill RUN_DIRS"
  exit 1
fi

if [[ ${#CHECKPOINTS[@]} -gt 0 && ${#CHECKPOINTS[@]} -ne ${#RUN_DIRS[@]} ]]; then
  log "ERROR: CHECKPOINTS (${#CHECKPOINTS[@]}) 须与 RUN_DIRS (${#RUN_DIRS[@]}) 等长，或全部留空。"
  tty_endl "evaluation FAILED — length mismatch"
  exit 1
fi

n=${#RUN_DIRS[@]}
for ((i = 0; i < n; i++)); do
  run_dir="${RUN_DIRS[$i]}"
  if [[ ! -d "$run_dir" ]]; then
    log "ERROR: run_dir 不存在: $run_dir"
    tty_endl "evaluation FAILED — missing run_dir"
    exit 1
  fi

  args=(
    -m llm4rec.cli.main evaluate
    "run_dir=${run_dir}"
    "experiment=${EXPERIMENT}"
    "checkpoint_stage=${CHECKPOINT_STAGE}"
  )

  if [[ ${#CHECKPOINTS[@]} -gt 0 ]]; then
    ckpt="${CHECKPOINTS[$i]}"
    if [[ ! -d "$ckpt" ]]; then
      log "ERROR: checkpoint 不存在: $ckpt"
      tty_endl "evaluation FAILED — missing checkpoint"
      exit 1
    fi
    args+=("adapter_path=${ckpt}")
    sft_guess="${run_dir}/checkpoints/sft/final"
    if [[ "$CHECKPOINT_STAGE" == "grpo" && -d "$sft_guess" ]]; then
      args+=("sft_adapter_path=${sft_guess}")
    fi
  fi

  log ""
  log "[$((i + 1))/${n}] run_dir=$run_dir stage=$CHECKPOINT_STAGE"
  step "eval_$((i + 1))" python "${args[@]}"
  out="${run_dir}/eval/${CHECKPOINT_STAGE}_metrics.json"
  if [[ -f "$out" ]]; then
    log "metrics → $out"
  fi
done

log ""
log "evaluation DONE  n=${n}  stage=${CHECKPOINT_STAGE}"
tty_endl "evaluation DONE n=${n} — see $LOG"
