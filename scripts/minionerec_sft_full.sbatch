#!/bin/bash
# minionerec task 3.3：full-scale SID prepare + SFT
#SBATCH -J minionerec_sft_full
#SBATCH -A aikyamlab
#SBATCH -p gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/minionerec_sft_full_%j.out
#SBATCH --error=logs/minionerec_sft_full_%j.err

set -euo pipefail
cd ~/llm4rec-bias-Integrated
mkdir -p logs
export PATH="$HOME/.conda/envs/bias/bin:$PATH"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export LLM4REC_COMPOSE="hardware=single scale=full"
export CUDA_VISIBLE_DEVICES=0

python -m llm4rec.cli.main validate workflow=minionerec scale=full
python -m llm4rec.cli.main prepare workflow=minionerec scale=full
python -m llm4rec.cli.main train workflow=minionerec scale=full stages=[sft]

echo "minionerec full SFT DONE"
