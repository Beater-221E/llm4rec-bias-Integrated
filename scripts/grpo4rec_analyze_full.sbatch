#!/bin/bash
# grpo4rec tasks 6.1–6.3：full-scale bias probes（default + subset）
#SBATCH -J grpo4rec_analyze
#SBATCH -A aikyamlab
#SBATCH -p gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/grpo4rec_analyze_%j.out
#SBATCH --error=logs/grpo4rec_analyze_%j.err

set -euo pipefail
cd ~/llm4rec-bias-Integrated
mkdir -p logs
# Rivanna sbatch：bias 在 ~/.conda/envs/bias
export PATH="$HOME/.conda/envs/bias/bin:$PATH"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export LLM4REC_COMPOSE="hardware=single scale=full"
export CUDA_VISIBLE_DEVICES=0

RUN_DIR="${RUN_DIR:-$(ls -dt runs/movielens_100k/grpo4rec/*/seed_*/20* 2>/dev/null | head -1 || true)}"
if [[ -z "${RUN_DIR:-}" || ! -d "$RUN_DIR" ]]; then
  echo "ERROR: no grpo4rec run_dir found" >&2
  exit 1
fi
echo "run_dir=$RUN_DIR"

python -m llm4rec.cli.main validate workflow=grpo4rec scale=full

# 6.1 default probes: popularity, position, framing, recency
python -m llm4rec.cli.main analyze workflow=grpo4rec scale=full "run_dir=${RUN_DIR}"
rm -rf "${RUN_DIR}/eval/probes_default"
cp -a "${RUN_DIR}/eval/probes" "${RUN_DIR}/eval/probes_default"

# 6.2 verify each default probe has numeric scores
python - <<PY
import json
from pathlib import Path

summary = json.loads(Path("${RUN_DIR}/eval/probes_default/summary.json").read_text())
expected = {"popularity", "position", "framing", "recency"}
ran = {p["name"] for p in summary.get("probes", [])}
missing = expected - ran
if missing:
    raise SystemExit(f"missing probes: {missing}")
for entry in summary["probes"]:
    name = entry["name"]
    metrics = entry.get("metrics") or {}
    if not metrics:
        raise SystemExit(f"probe {name} has empty metrics")
    for key, val in metrics.items():
        if not isinstance(val, (int, float)):
            raise SystemExit(f"probe {name} metric {key} not numeric: {val!r}")
print("OK: all default probes have numeric scores:", sorted(ran))
PY

# 6.3 subset probes only
python -m llm4rec.cli.main analyze workflow=grpo4rec scale=full \
  "run_dir=${RUN_DIR}" 'bias.probes=[popularity,recency]'
python - <<PY
import json
from pathlib import Path

summary = json.loads(Path("${RUN_DIR}/eval/probes/summary.json").read_text())
ran = [p["name"] for p in summary.get("probes", [])]
if ran != ["popularity", "recency"]:
    raise SystemExit(f"expected subset [popularity, recency], got {ran}")
Path("${RUN_DIR}/eval/probes_subset").mkdir(parents=True, exist_ok=True)
for name in ran:
    src = Path("${RUN_DIR}/eval/probes") / f"{name}.json"
    dst = Path("${RUN_DIR}/eval/probes_subset") / f"{name}.json"
    dst.write_text(src.read_text())
(Path("${RUN_DIR}/eval/probes_subset/summary.json")).write_text(
    Path("${RUN_DIR}/eval/probes/summary.json").read_text()
)
print("OK: subset probes only:", ran)
PY

echo "grpo4rec bias analysis DONE run_dir=$RUN_DIR"
