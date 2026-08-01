#!/bin/bash
# mllm4rec 冒烟续跑：Ranker only（retriever 已完成时用）
#SBATCH -J mllm4rec_rank_smoke
#SBATCH -A aikyamlab
#SBATCH -p gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/mllm4rec_rank_smoke_%j.out
#SBATCH --error=logs/mllm4rec_rank_smoke_%j.err

set -euo pipefail
cd ~/llm4rec-bias-Integrated
mkdir -p logs
export PATH="$HOME/.conda/envs/bias/bin:$PATH"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

EXPORT_LRU="experiments/lru/ml-100k-smoke"
EXPORT_RANKER="experiments/ranker/ml-100k-smoke"

test -f "$EXPORT_LRU/retrieved.pkl"

python -m llm4rec_bias_Integrated.mllm4rec.cli train-ranker \
  --config mllm4rec_ranker \
  --retrieved-pkl "$EXPORT_LRU/retrieved.pkl" \
  --export-root "$EXPORT_RANKER" \
  --max-train-steps 20 --num-epochs 1

test -f "$EXPORT_RANKER/subset_metrics.json"
echo "mllm4rec RANKER SMOKE DONE"
