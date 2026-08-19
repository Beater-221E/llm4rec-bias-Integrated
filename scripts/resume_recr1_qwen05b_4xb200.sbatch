#!/bin/bash
# 用途：用 4×B200 从已完成 SFT 恢复 Rec-R1，运行全量 eval→RL→eval。
#SBATCH -J resume_recr1_4g
#SBATCH -A sds-rcnode-1
#SBATCH -p dedicated
#SBATCH --reservation=sds-rcnode-1-2
#SBATCH --gres=gpu:b200:4
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/resume_recr1_4g_%j.out
#SBATCH --error=logs/resume_recr1_4g_%j.err

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

SFT_FINAL="/scratch/esd4uq/llm4rec-bias-Integrated/runs/amazon23_Industrial_and_Scientific/recr1/qwen2.5-0.5b-instruct/seed_42/20260813_005844/sft/final"
EXP=recr1_qwen05b_amazon STAGES=eval,rl,eval RESUME_FROM="$SFT_FINAL" \
  GPUS=0,1,2,3 bash run.sh mode=integrated
