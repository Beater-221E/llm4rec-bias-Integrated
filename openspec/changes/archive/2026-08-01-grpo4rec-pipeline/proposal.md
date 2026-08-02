## Why

The grpo4rec track establishes an LLM-based recommendation pipeline using Group Relative Policy Optimization (GRPO) on the classic MovieLens-100K dataset. It provides a supervised fine-tuning (SFT) warm-up stage followed by GRPO training to optimize for accurate candidate item selection, with built-in evaluation and bias analysis probes. This is the foundational track that validates the LLM-for-recommendation approach end-to-end.

## What Changes

- **Data preparation**: Download and preprocess classic MovieLens-100K (`u.data` / `u.item`) into train/dev/test splits with interaction histories, popularity stats, and framing metadata
- **SFT training stage**: LoRA fine-tune Qwen2.5-0.5B-Instruct on formatted recommendation prompts — teaches the model the candidate-choice task format
- **GRPO training stage**: Policy optimization with reward components (exact match, format validity, rank-aware, popularity penalty) — refines the model to prefer correct and diverse recommendations
- **Evaluation suite**: Ranking metrics (HR, Recall, NDCG, MRR at K=1/5/10) on held-out users
- **Bias analysis**: Popularity, position, framing, and recency bias probes
- **Two scale presets**: `smoke` (64 train / 32 eval, 4 steps) for rapid iteration; `full` for complete training runs

## Capabilities

### New Capabilities

- `data-preparation`: Ingest classic MovieLens-100K raw files, split users, construct interaction sequences, and persist to `data/processed/movielens_100k/`
- `grpo-training`: SFT warm-up → GRPO policy optimization pipeline with configurable LoRA, reward weights, and generation parameters
- `model-evaluation`: Ranking evaluation (HR, Recall, NDCG, MRR at top-K) on test splits, supporting both smoke and full scales
- `bias-analysis`: Probe-based bias measurement across popularity, position, framing, and recency dimensions

### Modified Capabilities

_No existing capabilities are modified — this is a new project._

## Impact

- **Config**: `config.yaml` → `grpo4rec` section (dataset, model, PEFT, SFT, GRPO, evaluation, analysis, scale presets)
- **Code**: `src/llm4rec/` — data loading, training loops, reward computation, evaluation, bias probes
- **CLI**: `python -m llm4rec.cli.main prepare|train|evaluate|analyze` with `workflow=grpo4rec`
- **Data**: `data/raw/movielens_100k/` → `data/processed/movielens_100k/`
- **Scripts**: `smoke.sh` (three-track orchestration), `grpo4rec.sh` (single-track convenience)
