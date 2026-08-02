## Why

The minionerec track extends the grpo4rec pipeline with Semantic ID (SID) tokens — hierarchical discrete codes that represent items for more efficient and structured generation. Instead of generating raw item identifiers, the model learns to predict 3-level residual K-means codes from a shared codebook (size 64), reducing the generation space and enabling collision-aware prediction.

## What Changes

- **Semantic ID generation**: On top of the same MovieLens-100K processed data, build a residual K-means codebook and map every item to a 3-level SID (`data/processed/movielens_100k/sid/`)
- **SID-aware training**: SFT and GRPO stages that format prompts and targets using SID token sequences instead of raw item text — the model learns to generate SID codes that decode to items
- **Collision handling**: Extra SID levels and collision-aware reward penalties to handle items that map to the same code
- **SID-specific evaluation**: Ranking metrics plus free-generation evaluation (SID validity, generation accuracy, semantic collision rates)
- **Two scale presets**: `smoke` (64 train / 16 eval, 4 steps) and `full` for complete runs

## Capabilities

### New Capabilities

- `sid-generation`: Build residual K-means codebook (64 × 3 levels) from item embeddings, produce `sid_*.jsonl` mapping items to SID sequences and the reverse codebook
- `sid-training`: SFT + GRPO training stages that use SID-tokenized prompts and decode generated SIDs back to item predictions — reuses grpo4rec's training infrastructure with SID-specific formatting
- `sid-evaluation`: Evaluate ranking metrics and free-form generation quality (SID validity, collision rate) on the SID-augmented test set

### Modified Capabilities

_No existing capabilities are modified — this is a new project._

## Impact

- **Config**: `config.yaml` → `minionerec` section (SID method/defaults, dataset reuse, training stages, evaluation, scale presets)
- **Code**: `src/llm4rec/` — SID codebook construction, SID formatting/decoding, collision-aware reward
- **CLI**: `python -m llm4rec.cli.main prepare experiment=smoke_sid` for SID prep; `train experiment=smoke_sid` for training
- **Data**: `data/processed/movielens_100k/sid/` — codebook, `sid_train.jsonl`, `sid_val.jsonl`, `sid_test.jsonl`
- **Scripts**: `minionerec.sh` (single-track convenience); included in `smoke.sh`
