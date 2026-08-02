#!/bin/bash
# minionerec task 5.5：full-scale SID eval（free_gen_n=50 + top-K）on latest GRPO/SFT
#SBATCH -J minionerec_eval_full
#SBATCH -A aikyamlab
#SBATCH -p gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/minionerec_eval_full_%j.out
#SBATCH --error=logs/minionerec_eval_full_%j.err

set -euo pipefail
cd ~/llm4rec-bias-Integrated
mkdir -p logs
export PATH="$HOME/.conda/envs/bias/bin:$PATH"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export LLM4REC_COMPOSE="hardware=single scale=full"
export CUDA_VISIBLE_DEVICES=0

GRPO_ADAPTER="${GRPO_ADAPTER:-$(ls -dt runs/movielens_100k/minionerec/*/seed_*/20*/checkpoints/grpo/final 2>/dev/null | head -1 || true)}"
if [[ -z "${GRPO_ADAPTER:-}" || ! -f "$GRPO_ADAPTER/adapter_config.json" ]]; then
  echo "ERROR: no minionerec GRPO adapter found" >&2
  exit 1
fi
# checkpoints/grpo/final → run_dir
RUN_DIR="${RUN_DIR:-$(cd "$(dirname "$GRPO_ADAPTER")/../.." && pwd)}"

SFT_ADAPTER="${SFT_ADAPTER:-}"
if [[ -z "${SFT_ADAPTER}" ]]; then
  if [[ -f "$RUN_DIR/checkpoints/sft/final/adapter_config.json" ]]; then
    SFT_ADAPTER="$RUN_DIR/checkpoints/sft/final"
  else
    SFT_ADAPTER="$(ls -dt runs/movielens_100k/minionerec/*/seed_*/20*/checkpoints/sft/final 2>/dev/null | head -1 || true)"
  fi
fi
if [[ -z "${SFT_ADAPTER:-}" || ! -f "$SFT_ADAPTER/adapter_config.json" ]]; then
  echo "ERROR: no SFT adapter found (needed to merge before GRPO eval)" >&2
  exit 1
fi

echo "run_dir=$RUN_DIR"
echo "sft_adapter=$SFT_ADAPTER"
echo "grpo_adapter=$GRPO_ADAPTER"

python -m llm4rec.cli.main validate workflow=minionerec scale=full

python - <<PY
from pathlib import Path

from llm4rec.core.config import load_config, validate_config
from llm4rec.core.paths import project_root
from llm4rec.core.reproducibility import write_json
from llm4rec.components.evaluation._impl.sid import evaluate_sid_checkpoint

run_dir = Path("${RUN_DIR}").resolve()
sft = Path("${SFT_ADAPTER}").resolve()
grpo = Path("${GRPO_ADAPTER}").resolve()
cfg = validate_config(load_config(["workflow=minionerec", "scale=full"]))
eval_cfg = cfg.get("evaluation") or {}
top_k = int((eval_cfg.get("top_k") or [10])[-1])
free_gen_n = int(eval_cfg.get("free_gen_n", 50))
max_examples = eval_cfg.get("max_examples")
processed = project_root() / "data" / "processed" / "movielens_100k"

# SFT-only baseline
sft_result = evaluate_sid_checkpoint(
    model_cfg=cfg["model"],
    processed_dir=processed,
    adapter_path=str(sft),
    sft_adapter_path=None,
    split="test",
    top_k=top_k,
    max_examples=max_examples,
    free_gen_n=free_gen_n,
    predictions_dir=run_dir / "predictions",
)
write_json(
    run_dir / "eval" / "test_sft_metrics.json",
    {
        "metrics": sft_result.metrics,
        "slices": sft_result.slices,
        "metadata": sft_result.metadata,
        "adapter_path": str(sft),
        "free_gen_n": free_gen_n,
    },
)
print("SFT test metrics:", sft_result.metrics)

# GRPO (merge SFT then GRPO adapter)
grpo_result = evaluate_sid_checkpoint(
    model_cfg=cfg["model"],
    processed_dir=processed,
    adapter_path=str(grpo),
    sft_adapter_path=str(sft),
    split="test",
    top_k=top_k,
    max_examples=max_examples,
    free_gen_n=free_gen_n,
    predictions_dir=run_dir / "predictions",
)
write_json(
    run_dir / "eval" / "test_grpo_metrics.json",
    {
        "metrics": grpo_result.metrics,
        "slices": grpo_result.slices,
        "metadata": grpo_result.metadata,
        "adapter_path": str(grpo),
        "sft_adapter_path": str(sft),
        "free_gen_n": free_gen_n,
    },
)
print("GRPO test metrics:", grpo_result.metrics)
print("Wrote", run_dir / "eval" / "test_sft_metrics.json")
print("Wrote", run_dir / "eval" / "test_grpo_metrics.json")
PY

echo "minionerec full eval DONE run_dir=$RUN_DIR"
