#!/usr/bin/env bash
# 一键冒烟：环境检查 + grpo4rec / minionerec / mllm4rec 限步跑通。
# 日志：logs/smoke.txt（每次覆盖）；终端只显示单行进度。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
LOG="$ROOT/logs/smoke.txt"
# shellcheck disable=SC1091
source "$ROOT/scripts/run_lib.sh"
activate_bias

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export LLM4REC_COMPOSE="hardware=single scale=smoke"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NVML_ENABLE="${NCCL_NVML_ENABLE:-0}"

log "smoke start → $LOG"
log "Python=$(command -v python)  COMPOSE=$LLM4REC_COMPOSE"
tty_endl "smoke → $LOG"

step env_cuda python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.device_count(), 'GPU(s)', torch.cuda.get_device_name(0))"
step validate_grpo python -m llm4rec_bias_Integrated.cli.main validate experiment=smoke_grpo
step validate_sid  python -m llm4rec_bias_Integrated.cli.main validate experiment=smoke_sid

step grpo4rec_prepare python -m llm4rec_bias_Integrated.cli.main prepare experiment=smoke_test dataset=movielens_100k
step grpo4rec_train   python -m llm4rec_bias_Integrated.cli.main train experiment=smoke_grpo

step minionerec_prepare python -m llm4rec_bias_Integrated.cli.main prepare experiment=smoke_sid
step minionerec_train   python -m llm4rec_bias_Integrated.cli.main train experiment=smoke_sid

PKL=data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/dataset.pkl
if [[ ! -f "$PKL" ]]; then
  step mllm_build python -m llm4rec_bias_Integrated.data.mllm4rec.cli build \
    --config configs/dataset/mllm4rec_ml100k.yaml --skip-multimodal
  step mllm_stub_captions python - "$PKL" <<'PY'
import pickle, sys
from pathlib import Path
p = Path(sys.argv[1])
ds = pickle.load(p.open("rb"))
ds["meta_img_des"] = {k: "" for k in (ds.get("meta") or {})}
tmp = p.with_suffix(".pkl.tmp")
pickle.dump(ds, tmp.open("wb"))
tmp.replace(p)
print("stubbed meta_img_des", len(ds["meta_img_des"]))
PY
fi
step mllm_retriever python -m llm4rec_bias_Integrated.mllm4rec.cli train-retriever \
  --config configs/training/mllm4rec_retriever.yaml --num-epochs 2 \
  --export-root experiments/lru/ml-100k-smoke
step mllm_ranker python -m llm4rec_bias_Integrated.mllm4rec.cli train-ranker \
  --config configs/training/mllm4rec_ranker.yaml \
  --retrieved-pkl experiments/lru/ml-100k-smoke/retrieved.pkl \
  --export-root experiments/ranker/ml-100k-smoke \
  --max-train-steps 20 --num-epochs 1

log ""
log "SMOKE PASSED — grpo4rec + minionerec + mllm4rec"
tty_endl "SMOKE PASSED — see $LOG"
