#!/usr/bin/env bash
# =============================================================================
# 一次性数据准备 —— 跑训练之前先跑这个，跑完就不用再跑了。
#
#   0. 下载原始数据      → data/raw/<dataset>/               （带断点续传）
#   1. 预处理            → data/processed/<dataset>/<变体>/   （统一四件套契约）
#   2. 编码物品文本      → artifacts/embeddings/<dataset>/<encoder>/
#   3. 构建 Semantic ID  → artifacts/sid/<dataset>/<hash>/    ★ 静态产物
#   4. 构建 BM25 索引    → artifacts/bm25/<dataset>/          （Rec-R1 路线要用）
#
# ★ SID 是静态的：只在这里生成一次，训练和评测全程【只读】。
#   训练启动时会校验 manifest 里的 config hash，对不上直接报错退出，
#   绝不会像旧版那样"发现缺了就偷偷重建"——那会让不同 run 的 SID 悄悄不一致。
# =============================================================================
set -euo pipefail

# ───────────────────────────── 配置区 ─────────────────────────────
EXP="${EXP:-minionerec}"          # 用哪个实验配置里的 data/ 和 sid/ 设定
GPUS="${GPUS:-0}"                 # 编码 + RQ-VAE 训练用哪张卡（单卡够）
STEPS="${STEPS:-download,data,embed,sid,bm25}"   # 想跳过某步就删掉对应项
FORCE="${FORCE:-0}"               # 1 = 即使产物已存在也重建
CONDA_ENV="${CONDA_ENV:-bias}"
# ─────────────────────────────────────────────────────────────────

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
LOG="logs/prepare_${TS}.log"
log() { printf '%s\n' "$*" | tee -a "$LOG"; }

if [[ -n "$CONDA_ENV" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM=false

ARGS=(--config "$EXP")
[[ "$FORCE" == "1" ]] && ARGS+=(--force)
ARGS+=("$@")

log "═══════════════════════════════════════════════"
log " 实验配置 : $EXP"
log " 步骤     : $STEPS"
log " 显卡     : $GPUS   强制重建: $FORCE"
log " 日志     : $LOG"
log "═══════════════════════════════════════════════"

IFS=',' read -ra STEP_LIST <<<"$STEPS"
for step in "${STEP_LIST[@]}"; do
  step="$(echo "$step" | xargs)"
  [[ -z "$step" ]] && continue
  log ""
  log "──── [$step] ────"
  case "$step" in
    download) python -m llm4rec.cli.main download-data "${ARGS[@]}" 2>&1 | tee -a "$LOG" ;;
    data)  python -m llm4rec.cli.main prepare-data  "${ARGS[@]}" 2>&1 | tee -a "$LOG" ;;
    embed) python -m llm4rec.cli.main embed-items   "${ARGS[@]}" 2>&1 | tee -a "$LOG" ;;
    sid)   python -m llm4rec.cli.main build-sid     "${ARGS[@]}" 2>&1 | tee -a "$LOG" ;;
    bm25)  python -m llm4rec.cli.main build-bm25    "${ARGS[@]}" 2>&1 | tee -a "$LOG" ;;
    *)     log "未知步骤 '$step'（可用：download / data / embed / sid / bm25）"; exit 1 ;;
  esac
done

log ""
log "准备完成 → $LOG"
