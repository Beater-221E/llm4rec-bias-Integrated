#!/bin/bash
# 用途：在 dedicated B200 上跑 Amazon23 Industrial 的 prepare（跳过 download）
#SBATCH -J prepare_amazon23
#SBATCH -A sds-rcnode-1
#SBATCH -p dedicated
#SBATCH --reservation=sds-rcnode-1-2
#SBATCH --gres=gpu:b200:1
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/prepare_amazon23_%j.out
#SBATCH --error=logs/prepare_amazon23_%j.err

set -euo pipefail
cd "$HOME/llm4rec-bias-Integrated"
mkdir -p logs

export PATH="$HOME/.conda/envs/bias/bin:$PATH"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export WANDB_MODE=online
export CONDA_ENV=bias
export CUDA_VISIBLE_DEVICES=0

EXP=minionerec_qwen05b_amazon STEPS=data,embed,sid,bm25 GPUS=0 bash prepare.sh
