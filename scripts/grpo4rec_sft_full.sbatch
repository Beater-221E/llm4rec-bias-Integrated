#!/bin/bash
# grpo4rec task 3.3：full-scale SFT（Qwen2.5-0.5B + LoRA，MovieLens-100K 全量）
#SBATCH -J grpo4rec_sft_full
#SBATCH -A aikyamlab
#SBATCH -p gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/grpo4rec_sft_full_%j.out
#SBATCH --error=logs/grpo4rec_sft_full_%j.err

set -euo pipefail
cd ~/llm4rec-bias-Integrated
mkdir -p logs
# Rivanna sbatch：bias 在 ~/.conda/envs/bias（login 上 conda activate bias 即可；批任务直接加 PATH）
export PATH="$HOME/.conda/envs/bias/bin:$PATH"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export LLM4REC_COMPOSE="hardware=single scale=full"
export CUDA_VISIBLE_DEVICES=0

python -m llm4rec.cli.main validate workflow=grpo4rec scale=full
python -m llm4rec.cli.main train workflow=grpo4rec scale=full stages=[sft]
