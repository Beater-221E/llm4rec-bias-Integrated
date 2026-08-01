#!/bin/bash
# mllm4rec task 3.3：full-scale Retriever（500 epochs, early stopping patience=20）
#SBATCH -J mllm4rec_ret_full
#SBATCH -A aikyamlab
#SBATCH -p gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/mllm4rec_retriever_full_%j.out
#SBATCH --error=logs/mllm4rec_retriever_full_%j.err

set -euo pipefail
cd ~/llm4rec-bias-Integrated
mkdir -p logs
export PATH="$HOME/.conda/envs/bias/bin:$PATH"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

PKL="data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/dataset.pkl"
EXPORT_LRU="experiments/lru/ml-100k"

if [[ ! -f "$PKL" ]]; then
  python -m llm4rec_bias_Integrated.data.mllm4rec.cli build \
    --config mllm4rec_ml100k --skip-multimodal
fi
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
  --config mllm4rec_retriever \
  --export-root "$EXPORT_LRU"

test -f "$EXPORT_LRU/retrieved.pkl"
test -f "$EXPORT_LRU/test_metrics.json"
echo "mllm4rec RETRIEVER FULL DONE"
