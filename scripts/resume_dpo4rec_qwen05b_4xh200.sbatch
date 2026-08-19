#!/bin/bash
# 用途：dedicated 4×H200 续跑 DPO4Rec 0.5B（复用 SFT+reranker；重采 pair → DPO → eval）
#SBATCH -J resume_dpo05b_4h
#SBATCH -A sds-rcnode-1
#SBATCH -p dedicated
#SBATCH --reservation=sds-rcnode-1
#SBATCH --gres=gpu:h200:4
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/resume_dpo05b_4h_%j.out
#SBATCH --error=logs/resume_dpo05b_4h_%j.err

set -euo pipefail
cd "$HOME/llm4rec-bias-Integrated"
mkdir -p logs

# 载入 wandb 密钥等私有环境变量。
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export PATH="$HOME/.conda/envs/bias/bin:$PATH"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export WANDB_MODE=online
export WANDB_PROJECT=llm4rec-bias
export WANDB_ENTITY="${WANDB_ENTITY:-dylan-li-dailin-university-of-virginia}"
export CONDA_ENV=bias
export HF_HOME=/scratch/esd4uq/llm4rec-bias-Integrated/.cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"

RUN_ROOT="/scratch/esd4uq/llm4rec-bias-Integrated/runs/amazon23_Industrial_and_Scientific/dpo4rec/qwen2.5-0.5b-instruct/seed_42/20260813_005525"
EXP=dpo4rec_qwen05b_amazon STAGES=eval,dpo,eval RESUME_FROM="$RUN_ROOT/sft/final" \
  GPUS=0,1,2,3 bash run.sh mode=integrated \
  "decoder.reranker.load_from=$RUN_ROOT/reranker" \
  wandb.enabled=true wandb.mode=online \
  wandb.project=llm4rec-bias \
  wandb.entity=dylan-li-dailin-university-of-virginia
