#!/usr/bin/env bash
# 一键冒烟：环境检查 + grpo4rec / minionerec / mllm4rec 限步跑通。
# 要求数据已就绪（见 README「数据集怎么建」）；不在此脚本里下载/全量多模态构建。
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

PROCESSED="$ROOT/data/processed/movielens_100k"
PKL="$ROOT/data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/dataset.pkl"

require_data() {
  local ok=1
  if [[ ! -d "$PROCESSED" ]] || [[ -z "$(find "$PROCESSED" -mindepth 1 ! -name '.gitkeep' 2>/dev/null | head -1)" ]]; then
    log "ERROR: missing Letter/SID processed data: $PROCESSED"
    log "  → python -m llm4rec.cli.main prepare experiment=smoke_test dataset=movielens_100k"
    ok=0
  fi
  if [[ ! -f "$PKL" ]]; then
    log "ERROR: missing MLLM dataset.pkl: $PKL"
    log "  → python -m llm4rec_bias_Integrated.data.mllm4rec.cli build --config mllm4rec_ml100k --skip-multimodal"
    ok=0
  fi
  if [[ "$ok" -ne 1 ]]; then
    tty_endl "FAIL smoke — prepare datasets first (see README)"
    exit 1
  fi
  log "OK data_ready processed=$PROCESSED pkl=$PKL"
  tty_endl "OK data_ready"
}

log "smoke start → $LOG"
log "Python=$(command -v python)  COMPOSE=$LLM4REC_COMPOSE"
tty_endl "smoke → $LOG"

step env_cuda python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.device_count(), 'GPU(s)', torch.cuda.get_device_name(0))"
require_data

step validate_grpo python -m llm4rec_bias_Integrated.cli.main validate experiment=smoke_grpo
step validate_sid  python -m llm4rec_bias_Integrated.cli.main validate experiment=smoke_sid

step grpo4rec_prepare python -m llm4rec_bias_Integrated.cli.main prepare experiment=smoke_test dataset=movielens_100k
step grpo4rec_train   python -m llm4rec_bias_Integrated.cli.main train experiment=smoke_grpo

step minionerec_prepare python -m llm4rec_bias_Integrated.cli.main prepare experiment=smoke_sid
step minionerec_train   python -m llm4rec_bias_Integrated.cli.main train experiment=smoke_sid

# MLLM：复用已有 pkl；若无 caption 则 stub 空描述（不调用 TMDB）
if ! python - "$PKL" >>"$LOG" 2>&1 <<'PY'
import pickle, sys
from pathlib import Path
ds = pickle.load(Path(sys.argv[1]).open("rb"))
ok = all(k in ds for k in ("train", "val", "test", "meta", "umap", "smap"))
caps = len(ds.get("meta_img_des") or {})
print(f"dataset_ok={ok} captions={caps}")
sys.exit(0 if ok and caps > 0 else 1)
PY
then
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
  --config mllm4rec_retriever --num-epochs 2 \
  --export-root experiments/lru/ml-100k-smoke
step mllm_ranker python -m llm4rec_bias_Integrated.mllm4rec.cli train-ranker \
  --config mllm4rec_ranker \
  --retrieved-pkl experiments/lru/ml-100k-smoke/retrieved.pkl \
  --export-root experiments/ranker/ml-100k-smoke \
  --max-train-steps 20 --num-epochs 1

log ""
log "SMOKE PASSED — grpo4rec + minionerec + mllm4rec"
tty_endl "SMOKE PASSED — see $LOG"
