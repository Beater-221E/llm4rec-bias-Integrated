#!/bin/bash
# minionerec task 4.5：从最新 full SFT adapter 运行 100-step full GRPO
#SBATCH -J minionerec_grpo_full
#SBATCH -A aikyamlab
#SBATCH -p gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/minionerec_grpo_full_%j.out
#SBATCH --error=logs/minionerec_grpo_full_%j.err

set -euo pipefail
cd ~/llm4rec-bias-Integrated
mkdir -p logs
export PATH="$HOME/.conda/envs/bias/bin:$PATH"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export LLM4REC_COMPOSE="hardware=single scale=full"
export CUDA_VISIBLE_DEVICES=0

SFT_ADAPTER="${SFT_ADAPTER:-$(ls -dt runs/movielens_100k/minionerec/*/seed_*/20*/checkpoints/sft/final 2>/dev/null | head -1 || true)}"
if [[ -z "${SFT_ADAPTER:-}" || ! -f "$SFT_ADAPTER/adapter_config.json" ]]; then
  echo "ERROR: no minionerec SFT adapter found" >&2
  exit 1
fi
echo "sft_adapter=$SFT_ADAPTER"

python -m llm4rec.cli.main validate workflow=minionerec scale=full \
  'training.stages=[grpo]' "init_checkpoint=${SFT_ADAPTER}"
python -m llm4rec.cli.main train workflow=minionerec scale=full \
  'training.stages=[grpo]' "init_checkpoint=${SFT_ADAPTER}"

echo "minionerec full GRPO DONE"
