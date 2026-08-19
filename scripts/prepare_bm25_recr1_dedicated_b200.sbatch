#!/bin/bash
# 用途：为 Rec-R1 构建 Amazon23 Industrial 的 BM25 索引（CPU；走 dedicated 以免另开队列）
#SBATCH -J prepare_bm25
#SBATCH -A sds-rcnode-1
#SBATCH -p dedicated
#SBATCH --reservation=sds-rcnode-1-2
#SBATCH --gres=gpu:b200:1
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/prepare_bm25_%j.out
#SBATCH --error=logs/prepare_bm25_%j.err

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

EXP=recr1_qwen05b_amazon STEPS=bm25 GPUS=0 bash prepare.sh
