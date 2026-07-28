#!/usr/bin/env bash
# Single-process or auto multi-GPU training via the bias conda env.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f /opt/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate bias
fi

EXTRA=("$@")
if [[ ${#EXTRA[@]} -eq 0 ]]; then
  EXTRA=(experiment=smoke_sft)
fi

echo "Python: $(which python)"
echo "Running: python -m llm4rec_bias_Integrated.cli.main train ${EXTRA[*]}"
python -m llm4rec_bias_Integrated.cli.main train "${EXTRA[@]}"
