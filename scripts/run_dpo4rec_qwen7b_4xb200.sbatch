#!/bin/bash
# 用途：dedicated 4×B200 跑 DPO4Rec Qwen2.5-7B（reranker+SFT DDP → eval → DPO → eval）
#SBATCH -J run_dpo4rec_7b_4g
#SBATCH -A sds-rcnode-1
#SBATCH -p dedicated
#SBATCH --reservation=sds-rcnode-1-2
#SBATCH --gres=gpu:b200:4
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/run_dpo4rec_7b_4g_%j.out
#SBATCH --error=logs/run_dpo4rec_7b_4g_%j.err

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

EXP=dpo4rec_qwen7b_amazon GPUS=0,1,2,3 \
  bash run.sh mode=integrated hardware.devices=0,1,2,3 \
  wandb.enabled=true wandb.mode=online \
  wandb.project=llm4rec-bias \
  wandb.entity=dylan-li-dailin-university-of-virginia
