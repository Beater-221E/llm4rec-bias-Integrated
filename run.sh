#!/usr/bin/env bash
# =============================================================================
# llm4rec-bias-Integrated —— 唯一训练/评测入口
#
# 一键启动：
#     EXP=minionerec_qwen05b_amazon GPUS=0,1,2,3 bash run.sh
#
# 有哪些实验可跑：
#     python -m llm4rec.cli.main list
#
# 临时改任意配置（不用动文件）：
#     bash run.sh train.sft.learning_rate=2e-5 train.rl.eval_steps=10
#
# 结果落在哪：
#     runs/<数据集>/<route>/<模型>/seed_<seed>/<时间戳>/
#       ├── resolved_config.json    这次跑用的完整配置（可直接复现）
#       ├── environment.json        git commit / 依赖版本 / 硬件
#       ├── metrics.jsonl           全部指标（wandb 挂了也不丢）
#       ├── summary.json            各 stage 的 checkpoint 路径 + 最终指标
#       ├── sft/final/              SFT 完的全参权重
#       ├── rl/final/               RL 完的全参权重
#       └── eval/
#           ├── eval_1.json         SFT 后的 bias 基线
#           ├── eval_2.json         RL 后的 bias
#           └── bias_delta.json     ★ 两者差值 = "RL 放大了多少 bias"
# =============================================================================
set -euo pipefail

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                              配 置 区                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# 跑哪条路线：minionerec | recr1 | dpo4rec
#   对应 configs/exp/<EXP>.yaml
# ★ 必须显式设置，例如：
#     EXP=minionerec_qwen05b_amazon GPUS=auto bash run.sh
EXP="${EXP:-}"

# 跑哪些阶段。留空 = 用 configs/exp/<EXP>.yaml 里写的 stages。
#   minionerec: sft,eval,rl,eval        （SFT→评→RL→评，全自动串起来）
#   recr1     : sft,eval,rl,eval        （原文无 SFT，我们统一加以对齐基线）
#   dpo4rec   : train_reranker,sft,eval,dpo,eval
# 只想单独评测已有 checkpoint 就写 STAGES="eval" 再配 RESUME_FROM。
STAGES="${STAGES:-}"

# 用哪几张卡（逗号分隔）。多卡会自动起 torchrun。
#   GPUS=auto  → 不限制 CUDA_VISIBLE_DEVICES，检测全部可见 GPU
#   GPUS=0     → 单卡
#   GPUS=0,2   → 仅使用指定卡
GPUS="${GPUS:-auto}"

# DeepSpeed（只作用于 SFT 阶段）：留空 = 不用，用 DDP
#   zero2 | zero2_offload | zero3   → configs/deepspeed/<name>.yaml
#   0.5B 全参单卡放得下，留空即可；3B+ 建议 zero2；7B+ 用 zero3
#   ★ RL 阶段不走 DeepSpeed —— generate 在 ZeRO-3 下每步都要 gather 参数，会极慢
DEEPSPEED="${DEEPSPEED:-}"

# 从已有 checkpoint 继续 / 单独评测。留空 = 从 backbone 开始。
RESUME_FROM="${RESUME_FROM:-}"

# —— wandb ——
WANDB_MODE="${WANDB_MODE:-online}"          # online | offline | disabled
WANDB_PROJECT="${WANDB_PROJECT:-llm4rec-bias}"
WANDB_ENTITY="${WANDB_ENTITY:-}"            # 你的 team；留空用个人默认
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"        # 留空自动生成 <exp>_<model>_seed<seed>_<ts>

# —— 临时覆盖任意配置项（dotted key）——
# 例：OVERRIDES=( "train.rl.max_steps=100" "train.rl.bias_eval_steps=20" )
OVERRIDES=(
)

# conda 环境名（留空 = 不切换）
CONDA_ENV="${CONDA_ENV:-bias}"

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        以下一般不用改                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -z "${EXP}" ]]; then
  cat <<'EOF' >&2
ERROR: EXP is required.

Examples:
  EXP=minionerec_qwen05b_amazon GPUS=auto bash run.sh
  EXP=minionerec_reproduction_qwen05b GPUS=0,1 bash run.sh
  EXP=recr1_qwen05b_amazon STAGES=sft,eval GPUS=auto bash run.sh

List experiments:
  python -m llm4rec.cli.main list
EOF
  exit 1
fi

# 命令行尾部的 key=value 也当作 override
OVERRIDES+=("$@")

TS="$(date +%Y%m%d_%H%M%S)"
export LLM4REC_RUN_TS="$TS"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${EXP}_${TS}.log"

log() { printf '%s\n' "$*" | tee -a "$LOG"; }

# Persist console output to $LOG, but drop tqdm progress-only rows
# (e.g. " 61%|█| 6500/10728 [..]") which spam the file when stdout is not a TTY.
# Periodic Trainer metrics like "{'loss': ...}" are kept.
_tee_train_log() {
  # Match HF/tqdm progress bars; keep blank lines after metrics for readability
  local prog='^[[:space:]]*[0-9]+%\|'
  if command -v stdbuf >/dev/null 2>&1; then
    stdbuf -oL -eL tee >(stdbuf -oL grep -a -Ev "$prog" >>"$LOG")
  else
    tee >(grep -a --line-buffered -Ev "$prog" >>"$LOG")
  fi
}

# —— 环境 ——
if [[ -n "$CONDA_ENV" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    log "WARN: 找不到 conda，跳过环境切换（CONDA_ENV=$CONDA_ENV）"
  fi
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export WANDB_MODE="$WANDB_MODE"
export WANDB_PROJECT="$WANDB_PROJECT"
[[ -n "$WANDB_ENTITY" ]] && export WANDB_ENTITY="$WANDB_ENTITY"
[[ -n "$WANDB_RUN_NAME" ]] && export WANDB_NAME="$WANDB_RUN_NAME"

# devices:auto → do not restrict CUDA_VISIBLE_DEVICES; otherwise pin to GPUS
if [[ "$GPUS" == "auto" ]]; then
  unset CUDA_VISIBLE_DEVICES || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    NGPU="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  else
    NGPU=0
  fi
  if [[ -z "$NGPU" || "$NGPU" -lt 1 ]]; then
    NGPU=1
  fi
  log "GPUS=auto → using all visible devices (n=$NGPU)"
else
  export CUDA_VISIBLE_DEVICES="$GPUS"
  NGPU="$(awk -F',' '{print NF}' <<<"$GPUS")"
fi

# 多卡 NCCL：
#   - Ampere+ / NVLink：默认拓扑自检（可设 LLM4REC_NCCL_COMPAT=1 强制兼容档）
#   - 预 Ampere（V100 等 cc<8）：默认开兼容档，否则 P2P barrier/allreduce 易死锁
#   - 显式 LLM4REC_NCCL_COMPAT=0 可关闭自动兼容
if (( NGPU > 1 )); then
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  _cc_major="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)"
  _auto_compat=0
  if [[ -n "${_cc_major}" && "${_cc_major}" -lt 8 ]]; then
    _auto_compat=1
  fi
  if [[ "${LLM4REC_NCCL_COMPAT:-}" == "1" || ( "${LLM4REC_NCCL_COMPAT:-}" != "0" && "${_auto_compat}" == "1" ) ]]; then
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    export NCCL_NVML_ENABLE="${NCCL_NVML_ENABLE:-0}"
    log "NCCL compat profile ON (P2P/IB disabled; cc=${_cc_major:-unknown})"
  else
    log "NCCL defaults: topology detection enabled (set LLM4REC_NCCL_COMPAT=1 to disable P2P/IB)"
  fi
fi

# —— 组装 CLI 参数 ——
ARGS=(--config "$EXP")
[[ -n "$STAGES" ]] && ARGS+=(--stages "$STAGES")
[[ -n "$RESUME_FROM" ]] && ARGS+=(--resume-from "$RESUME_FROM")
[[ -n "$DEEPSPEED" ]] && ARGS+=("hardware.deepspeed=$DEEPSPEED")
for ov in "${OVERRIDES[@]}"; do
  [[ -n "$ov" ]] && ARGS+=("$ov")
done

log "════════════════════════════════════════════════════════════"
log " 实验     : $EXP"
log " 阶段     : ${STAGES:-<配置文件默认>}"
log " 显卡     : $GPUS  (n=$NGPU)"
if (( NGPU > 1 )); then
  log " 并行     : SFT=${DEEPSPEED:-DDP}  RL/DPO=DDP"
else
  log " 并行     : 单卡"
fi
log " wandb    : $WANDB_MODE / $WANDB_PROJECT"
log " 覆盖项   : ${OVERRIDES[*]:-<无>}"
log " 日志     : $LOG"
log "════════════════════════════════════════════════════════════"

# —— 先做一次纯 CPU 的配置校验，别等加载完模型才发现配错 ——
python -m llm4rec.cli.main validate "${ARGS[@]}" 2>&1 | tee -a "$LOG"

# —— 训练 ——
# 单卡直接跑；多卡走 torchrun。stage 之间的串联（SFT→eval→RL→eval）在 CLI 内部完成，
# 所以这里只起一次进程，torchrun 的 N 个进程各自走完整条 stage 链。
#
# 各 stage 的并行方式：
#   SFT    → HF Trainer（DDP 或 DeepSpeed，取决于 DEEPSPEED）
#   RL/DPO → llm4rec.core.distributed：按 rank 分样本 + DDP all-reduce 梯度，
#            只有 rank0 写日志/存 checkpoint，stage 边界有 barrier
#   eval   → 各 rank 解码一片再 all-gather 汇总
if (( NGPU > 1 )); then
  MASTER_PORT="${MASTER_PORT:-$(( 20000 + RANDOM % 20000 ))}"
  log "多卡模式：torchrun --nproc_per_node $NGPU --master_port $MASTER_PORT"
  torchrun \
    --standalone \
    --nproc_per_node "$NGPU" \
    --master_port "$MASTER_PORT" \
    -m llm4rec.cli.main run "${ARGS[@]}" 2>&1 | _tee_train_log
else
  python -m llm4rec.cli.main run "${ARGS[@]}" 2>&1 | _tee_train_log
fi

log ""
log "完成 → $LOG"
