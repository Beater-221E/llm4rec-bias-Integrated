## Why

The mllm4rec track establishes a two-stage (Retriever → Ranker) multi-modal LLM-based recommendation pipeline on the newer GroupLens ml-latest-small dataset. Unlike the grpo4rec/minionerec tracks which use a single LLM with LoRA, mllm4rec follows the established sequential recommendation paradigm: a lightweight BERT-based retriever narrows the candidate space, then an LLM ranker (Qwen2.5) re-ranks candidates with full context (history + item metadata + images). This evaluates a complementary architecture for LLM-based recommendation.

## What Changes

- **Data preparation**: Process ml-latest-small interactions into a unified `dataset.pkl` — supports text-only mode (`--skip-multimodal`) for smoke tests and full multimodal mode (TMDB posters + BLIP2 captions) for production runs
- **Retriever training**: Train a SASRec-style BERT retriever (2 blocks, hidden 64) with early stopping — produces `retrieved.pkl` with top-K candidates per user
- **Ranker training**: LoRA fine-tune Qwen2.5-0.5B-Instruct to re-rank candidates from the retriever — uses negative sampling (19 negatives per positive) and item serialization with available metadata
- **Two-stage evaluation**: Retrieval recall metrics + ranking metrics on final ranked output
- **Two scale presets**: `smoke` (2 retriever epochs, 1 ranker epoch / 20 steps) and `full` (500 retriever epochs, 3 ranker epochs)

## Capabilities

### New Capabilities

- `multimodal-data-preparation`: Process ml-latest-small (optionally ml-1m) into train/val/test splits with item text, TMDB poster images, and BLIP2 captions — serialized to `dataset.pkl` and Parquet
- `retriever-training`: BERT-based candidate retrieval with early stopping, producing `retrieved.pkl` for the ranker
- `ranker-training`: LoRA-based LLM re-ranking of retrieved candidates with configurable negative sampling and history truncation
- `two-stage-evaluation`: End-to-end evaluation from retrieval recall through final ranking metrics

### Modified Capabilities

_No existing capabilities are modified — this is a new project._

## Impact

- **Config**: `config.yaml` → `mllm4rec` section (data for ml-100k and ml-1m, retriever, ranker, scale presets)
- **Code**: `src/llm4rec_bias_Integrated/mllm4rec/` — data CLI, retriever trainer, ranker trainer
- **CLI**: `python -m llm4rec_bias_Integrated.data.mllm4rec.cli build` (data); `train-retriever` / `train-ranker` (training)
- **Data**: `data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/dataset.pkl` + images + captions
- **Scripts**: `mllm4rec.sh` (single-track convenience); included in `smoke.sh`
