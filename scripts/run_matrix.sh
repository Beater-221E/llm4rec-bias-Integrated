#!/usr/bin/env bash
# Force multi-GPU Accelerate launch using the bias env interpreter.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f /opt/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate bias
fi

NPROC="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${NPROC}" -lt 1 ]]; then
  echo "ERROR: no CUDA GPUs visible" >&2
  exit 1
fi

EXTRA=("$@")
if [[ ${#EXTRA[@]} -eq 0 ]]; then
  # Manual Accelerate launch → keep auto_launch off to avoid double-spawn.
  EXTRA=(experiment=smoke_sft hardware=multi training.auto_launch_multi_gpu=false)
fi

echo "Python: $(which python)"
echo "Launching with ${NPROC} processes via python -m accelerate"
export LLM4REC_FULL_DISTRIBUTED_CHILD=1
export NCCL_NVML_ENABLE="${NCCL_NVML_ENABLE:-0}"
python -m accelerate.commands.launch --num_processes "${NPROC}" --multi_gpu \
  -m llm4rec_bias_Integrated.cli.main train "${EXTRA[@]}"
