#!/bin/bash
# grpo4rec task 7.2：跑 convenience script（单卡 smoke 流程）
#SBATCH -J grpo4rec_e2e
#SBATCH -A aikyamlab
#SBATCH -p gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/grpo4rec_e2e_%j.out
#SBATCH --error=logs/grpo4rec_e2e_%j.err

set -euo pipefail
cd ~/llm4rec-bias-Integrated
mkdir -p logs
export PATH="$HOME/.conda/envs/bias/bin:$PATH"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

HARDWARE=single ./grpo4rec.sh
echo "grpo4rec.sh DONE"
