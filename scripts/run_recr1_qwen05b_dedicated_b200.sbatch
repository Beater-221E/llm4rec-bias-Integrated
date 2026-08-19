#!/bin/bash
# 用途：dedicated 1×B200 跑 Rec-R1 integrated（SFT→eval→RL→eval）；须已有 BM25
#SBATCH -J run_recr1
#SBATCH -A sds-rcnode-1
#SBATCH -p dedicated
#SBATCH --reservation=sds-rcnode-1-2
#SBATCH --gres=gpu:b200:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/run_recr1_%j.out
#SBATCH --error=logs/run_recr1_%j.err

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

EXP=recr1_qwen05b_amazon GPUS=0 bash run.sh mode=integrated
