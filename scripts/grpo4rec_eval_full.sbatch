#!/bin/bash
# grpo4rec task 5.2：full-scale ranking eval on complete test split (SFT + GRPO)
#SBATCH -J grpo4rec_eval_full
#SBATCH -A aikyamlab
#SBATCH -p gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/grpo4rec_eval_full_%j.out
#SBATCH --error=logs/grpo4rec_eval_full_%j.err

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
  echo "ERROR: no grpo4rec run_dir found under runs/movielens_100k/grpo4rec/" >&2
  exit 1
fi
echo "run_dir=$RUN_DIR"

python -m llm4rec.cli.main validate workflow=grpo4rec scale=full

# SFT checkpoint → test split (evaluate CLI)
python -m llm4rec.cli.main evaluate workflow=grpo4rec scale=full \
  "run_dir=${RUN_DIR}" evaluation.split=test

# GRPO checkpoint (merged over SFT) → test split
python - <<PY
from pathlib import Path

from llm4rec.core.config import load_config, validate_config
from llm4rec.core.reproducibility import write_json
from llm4rec.components.evaluation._impl.runner import evaluate_checkpoint_on_examples
from llm4rec.cli.evaluate import _build_dataset
from llm4rec.workflows.grpo4rec import task_spec_from_config

run_dir = Path("${RUN_DIR}").resolve()
sft = run_dir / "checkpoints" / "sft" / "final"
grpo = run_dir / "checkpoints" / "grpo" / "final"
if not sft.is_dir() or not grpo.is_dir():
    raise SystemExit(f"missing adapters: sft={sft} grpo={grpo}")

cfg = validate_config(load_config(["workflow=grpo4rec", "scale=full"]))
eval_cfg = cfg.get("evaluation") or {}
dataset = _build_dataset(cfg)
dataset.preprocess()
examples = dataset.build_examples("test", task_spec_from_config(cfg))
top_k = list(eval_cfg.get("top_k") or [1, 5, 10])
use_upstream = bool(eval_cfg.get("use_upstream_eval", True))

result = evaluate_checkpoint_on_examples(
    model_cfg=cfg["model"],
    examples=examples,
    adapter_path=str(grpo),
    sft_adapter_path=str(sft),
    top_k=top_k,
    predictions_dir=run_dir / "predictions",
    split="test_grpo",
)
out = run_dir / "eval" / "test_grpo_metrics.json"
write_json(
    out,
    {
        "metrics": result.metrics,
        "slices": result.slices,
        "metadata": result.metadata,
        "use_upstream_eval": use_upstream,
        "adapter_path": str(grpo),
        "sft_adapter_path": str(sft),
    },
)
print("GRPO test metrics:", result.metrics)
print("Wrote", out)
PY

echo "grpo4rec full eval DONE run_dir=$RUN_DIR"
