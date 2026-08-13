#!/usr/bin/env bash
# Real multi-GPU trainer validation (GRPO / DPO / FSDP / FSDP ref-sync).
# Skips cleanly when fewer than 2 CUDA devices are visible.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

N="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "${N}" -lt 2 ]]; then
  echo "[validate_multi_gpu] need >=2 GPUs (have ${N}); skipping"
  exit 0
fi

MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
echo "[validate_multi_gpu] torchrun nproc=2 master_port=${MASTER_PORT}"
torchrun --standalone --nproc_per_node=2 --master_port "${MASTER_PORT}" \
  -m pytest \
  tests/test_trainer_dist_smoke.py \
  tests/test_fsdp_reference_sync.py \
  -q "$@"
