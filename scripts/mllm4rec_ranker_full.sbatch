#!/bin/bash
# mllm4rec task 4.4：full-scale Ranker（3 epochs, micro_batch_size=2, lr=1e-4）
#SBATCH -J mllm4rec_rank_full
#SBATCH -A aikyamlab
#SBATCH -p gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/mllm4rec_ranker_full_%j.out
#SBATCH --error=logs/mllm4rec_ranker_full_%j.err

set -euo pipefail
cd ~/llm4rec-bias-Integrated
mkdir -p logs
export PATH="$HOME/.conda/envs/bias/bin:$PATH"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

RETRIEVED="experiments/lru/ml-100k/retrieved.pkl"
EXPORT_RANKER="experiments/ranker/ml-100k"

if [[ ! -f "$RETRIEVED" ]]; then
  echo "Missing $RETRIEVED — run mllm4rec_retriever_full.sbatch first"
  exit 1
fi

python -m llm4rec_bias_Integrated.mllm4rec.cli train-ranker \
  --config mllm4rec_ranker \
  --retrieved-pkl "$RETRIEVED" \
  --export-root "$EXPORT_RANKER"

test -d "$EXPORT_RANKER/best_lora"
test -f "$EXPORT_RANKER/subset_metrics.json"
echo "mllm4rec RANKER FULL DONE"
