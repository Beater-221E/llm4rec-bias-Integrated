## Why

minionerec extends grpo4rec with Semantic ID (SID) tokens — hierarchical discrete codes representing items for more efficient, structured generation. Model learns 3-level residual K-means codes from shared codebook (size 64) instead of raw item identifiers, shrinking generation space + enabling collision-aware prediction.

## What Changes

- **Semantic ID generation**: On top of same MovieLens-100K processed data, build residual K-means codebook, map every item to 3-level SID (`data/processed/movielens_100k/sid/`)
- **SID-aware training**: SFT + GRPO stages formatting prompts/targets as SID token sequences instead of raw item text — model learns SID codes decoding to items
- **Collision handling**: Extra SID levels + collision-aware reward penalties for items mapping to same code
- **SID-specific evaluation**: Ranking metrics + free-generation evaluation (SID validity, generation accuracy, semantic collision rates)
- **Two scale presets**: `smoke` (64 train / 16 eval, 4 steps) and `full`

## Capabilities

### New Capabilities

- `sid-generation`: Build residual K-means codebook (64 × 3 levels) from item embeddings, produce `sid_*.jsonl` mapping items to SID sequences + reverse codebook
- `sid-training`: SFT + GRPO stages using SID-tokenized prompts, decoding generated SIDs to item predictions — reuses grpo4rec training infra with SID-specific formatting
- `sid-evaluation`: Evaluate ranking metrics + free-form generation quality (SID validity, collision rate) on SID-augmented test set

### Modified Capabilities

_No existing capabilities modified — new project._

## Impact

- **Config**: `config.yaml` → `minionerec` section (SID method/defaults, dataset reuse, training stages, evaluation, scale presets)
- **Code**: `src/llm4rec/` — SID codebook construction, SID formatting/decoding, collision-aware reward
- **CLI**: `python -m llm4rec.cli.main prepare experiment=smoke_sid` for SID prep; `train experiment=smoke_sid` for training
- **Data**: `data/processed/movielens_100k/sid/` — codebook, `sid_train.jsonl`, `sid_val.jsonl`, `sid_test.jsonl`
- **Scripts**: `minionerec.sh` (single-track convenience); in `smoke.sh`
