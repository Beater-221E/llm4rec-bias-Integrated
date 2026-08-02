#!/bin/bash
# minionerec 冒烟：SID prepare → SFT(4步) → GRPO(4步) → evaluate
#SBATCH -J minionerec_smoke
#SBATCH -A aikyamlab
#SBATCH -p gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/minionerec_smoke_%j.out
#SBATCH --error=logs/minionerec_smoke_%j.err

set -euo pipefail
cd ~/llm4rec-bias-Integrated
mkdir -p logs
# Rivanna sbatch：bias 在 ~/.conda/envs/bias
export PATH="$HOME/.conda/envs/bias/bin:$PATH"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export LLM4REC_COMPOSE="hardware=single scale=smoke"
export CUDA_VISIBLE_DEVICES=0

python -m llm4rec.cli.main validate experiment=smoke_sid
python -m llm4rec.cli.main prepare experiment=smoke_sid
python -m llm4rec.cli.main train experiment=smoke_sid
