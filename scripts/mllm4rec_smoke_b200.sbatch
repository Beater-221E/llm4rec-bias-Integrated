#!/bin/bash
# mllm4rec 冒烟：text-only data build → Retriever(2 epoch) → Ranker(1 epoch / 20 steps)
#SBATCH -J mllm4rec_smoke
#SBATCH -A aikyamlab
#SBATCH -p gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/mllm4rec_smoke_%j.out
#SBATCH --error=logs/mllm4rec_smoke_%j.err

set -euo pipefail
cd ~/llm4rec-bias-Integrated
mkdir -p logs
export PATH="$HOME/.conda/envs/bias/bin:$PATH"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

PKL="data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/dataset.pkl"
EXPORT_LRU="experiments/lru/ml-100k-smoke"
EXPORT_RANKER="experiments/ranker/ml-100k-smoke"

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.get_device_name(0))"

if [[ ! -f "$PKL" ]]; then
  python -m llm4rec_bias_Integrated.data.mllm4rec.cli build \
    --config mllm4rec_ml100k --skip-multimodal
fi
# Text-only pkl may lack meta_img_des; ranker needs stub captions (same as smoke.sh).
python - "$PKL" <<'PY'
import pickle, sys
from pathlib import Path
p = Path(sys.argv[1])
ds = pickle.load(p.open("rb"))
if "meta_img_des" not in ds:
    ds["meta_img_des"] = {k: "" for k in (ds.get("meta") or {})}
    tmp = p.with_suffix(".pkl.tmp")
    pickle.dump(ds, tmp.open("wb"))
    tmp.replace(p)
    print("stubbed meta_img_des", len(ds["meta_img_des"]))
PY

python -m llm4rec_bias_Integrated.mllm4rec.cli train-retriever \
  --config mllm4rec_retriever --num-epochs 2 \
  --export-root "$EXPORT_LRU"

python -m llm4rec_bias_Integrated.mllm4rec.cli train-ranker \
  --config mllm4rec_ranker \
  --retrieved-pkl "$EXPORT_LRU/retrieved.pkl" \
  --export-root "$EXPORT_RANKER" \
  --max-train-steps 20 --num-epochs 1

test -f "$EXPORT_LRU/retrieved.pkl"
test -f "$EXPORT_LRU/test_metrics.json"
test -f "$EXPORT_RANKER/subset_metrics.json"

echo "mllm4rec SMOKE DONE"
