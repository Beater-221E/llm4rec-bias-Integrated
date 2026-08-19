## Why

grpo4rec track builds LLM-based recommendation pipeline via Group Relative Policy Optimization (GRPO) on classic MovieLens-100K. SFT warm-up stage → GRPO training optimizing accurate candidate item selection, with built-in evaluation + bias analysis probes. Foundational track validating LLM-for-recommendation end-to-end.

## What Changes

- **Data preparation**: Download + preprocess classic MovieLens-100K (`u.data` / `u.item`) into train/dev/test splits with interaction histories, popularity stats, framing metadata
- **SFT training stage**: LoRA fine-tune Qwen2.5-0.5B-Instruct on formatted recommendation prompts — teaches candidate-choice task format
- **GRPO training stage**: Policy optimization with reward components (exact match, format validity, rank-aware, popularity penalty) — refines model toward correct + diverse recommendations
- **Evaluation suite**: Ranking metrics (HR, Recall, NDCG, MRR at K=1/5/10) on held-out users
- **Bias analysis**: Popularity, position, framing, recency bias probes
- **Two scale presets**: `smoke` (64 train / 32 eval, 4 steps) for rapid iteration; `full` for complete runs

## Capabilities

### New Capabilities

- `data-preparation`: Ingest classic MovieLens-100K raw files, split users, build interaction sequences, persist to `data/processed/movielens_100k/`
- `grpo-training`: SFT warm-up → GRPO policy optimization with configurable LoRA, reward weights, generation params
- `model-evaluation`: Ranking evaluation (HR, Recall, NDCG, MRR at top-K) on test splits, smoke + full scales
- `bias-analysis`: Probe-based bias measurement across popularity, position, framing, recency

### Modified Capabilities

_No existing capabilities modified — new project._

## Impact

- **Config**: `config.yaml` → `grpo4rec` section (dataset, model, PEFT, SFT, GRPO, evaluation, analysis, scale presets)
- **Code**: `src/llm4rec/` — data loading, training loops, reward computation, evaluation, bias probes
- **CLI**: `python -m llm4rec.cli.main prepare|train|evaluate|analyze` with `workflow=grpo4rec`
- **Data**: `data/raw/movielens_100k/` → `data/processed/movielens_100k/`
- **Scripts**: `smoke.sh` (three-track orchestration), `grpo4rec.sh` (single-track convenience)
