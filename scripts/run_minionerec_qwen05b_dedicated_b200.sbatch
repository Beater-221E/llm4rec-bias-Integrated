#!/bin/bash
# 用途：dedicated 1×B200 跑 MiniOneRec integrated（SFT→eval→RL→eval）
#SBATCH -J run_minionerec
#SBATCH -A sds-rcnode-1
#SBATCH -p dedicated
#SBATCH --reservation=sds-rcnode-1-2
#SBATCH --gres=gpu:b200:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/run_minionerec_%j.out
#SBATCH --error=logs/run_minionerec_%j.err

set -euo pipefail
cd "$HOME/llm4rec-bias-Integrated"
mkdir -p logs

export PATH="$HOME/.conda/envs/bias/bin:$PATH"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export WANDB_MODE=online
export CONDA_ENV=bias
export HF_HOME=/scratch/esd4uq/llm4rec-bias-Integrated/.cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"

EXP=minionerec_qwen05b_amazon GPUS=0 bash run.sh mode=integrated
