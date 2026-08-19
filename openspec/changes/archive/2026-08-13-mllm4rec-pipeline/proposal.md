## Why

mllm4rec establishes two-stage (Retriever → Ranker) multi-modal LLM recommendation on newer GroupLens ml-latest-small dataset. Unlike grpo4rec/minionerec (single LLM with LoRA), mllm4rec follows established sequential recommendation paradigm: lightweight BERT retriever narrows candidates, LLM ranker (Qwen2.5) re-ranks with full context (history + item metadata + images). Evaluates complementary architecture for LLM-based recommendation.

## What Changes

- **Data preparation**: Process ml-latest-small interactions into unified `dataset.pkl` — text-only mode (`--skip-multimodal`) for smoke tests + full multimodal (TMDB posters + BLIP2 captions) for production runs
- **Retriever training**: Train SASRec-style BERT retriever (2 blocks, hidden 64) with early stopping — produces `retrieved.pkl` with top-K candidates per user
- **Ranker training**: LoRA fine-tune Qwen2.5-0.5B-Instruct to re-rank retriever candidates — negative sampling (19 per positive) + item serialization with available metadata
- **Two-stage evaluation**: Retrieval recall metrics + ranking metrics on final output
- **Two scale presets**: `smoke` (2 retriever epochs, 1 ranker epoch / 20 steps) and `full` (500 retriever epochs, 3 ranker epochs)

## Capabilities

### New Capabilities

- `multimodal-data-preparation`: Process ml-latest-small (optionally ml-1m) into train/val/test splits with item text, TMDB posters, BLIP2 captions — serialized to `dataset.pkl` + Parquet
- `retriever-training`: BERT-based candidate retrieval with early stopping, producing `retrieved.pkl` for ranker
- `ranker-training`: LoRA-based LLM re-ranking of retrieved candidates, configurable negative sampling + history truncation
- `two-stage-evaluation`: End-to-end evaluation from retrieval recall through final ranking metrics

### Modified Capabilities

_No existing capabilities modified — new project._

## Impact

- **Config**: `config.yaml` → `mllm4rec` section (data for ml-100k + ml-1m, retriever, ranker, scale presets)
- **Code**: `src/llm4rec_bias_Integrated/mllm4rec/` — data CLI, retriever trainer, ranker trainer
- **CLI**: `python -m llm4rec_bias_Integrated.data.mllm4rec.cli build` (data); `train-retriever` / `train-ranker` (training)
- **Data**: `data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/dataset.pkl` + images + captions
- **Scripts**: `mllm4rec.sh` (single-track convenience); in `smoke.sh`
