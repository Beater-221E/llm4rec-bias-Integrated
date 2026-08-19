## Context

mllm4rec operates independently of grpo4rec/minionerec. Own codebase (`src/llm4rec_bias_Integrated/mllm4rec/`), own CLI, different dataset (ml-latest-small vs classic ML-100K). Two-stage paradigm common in sequential recommendation research: fast retriever narrows candidate space, powerful ranker (LLM) re-ranks with full context.

Data preprocessing = separate CLI (`build`): TMDB API calls, image downloads, BLIP2 captioning. Two separate training CLIs for retriever + ranker.

## Goals / Non-Goals

**Goals:**
- Two-stage evaluation: retriever must produce `retrieved.pkl` before ranker training
- Multimodal: item metadata = text, poster images, BLIP2 captions when available
- Text-only smoke mode: `--skip-multimodal` enables rapid testing without API dependencies

**Non-Goals:**
- Single-model recommendation — explicitly two-stage
- Integrate with grpo4rec/minionerec Hydra CLI — mllm4rec has own config system
- End-to-end training (joint retriever+ranker) — stages trained independently

## Decisions

### Two-stage over single LLM
**Decision**: Retriever (BERT, 2 blocks, hidden=64) → Ranker (Qwen2.5-0.5B, LoRA).
**Rationale**: Candidate space (1,700 items) too large for single LLM to score efficiently in one pass. BERT retriever O(n) over items, narrows to manageable top-K. LLM ranker then cross-attends with full context on much smaller set. Mirrors established RecSys architectures (SASRec + BERT reranker).
**Alternatives considered**: Single LLM listwise ranking (rejected: context window limits, VRAM); LLM as retriever (rejected: too slow for full-corpus candidate generation).

### BERT retriever over larger models
**Decision**: 2-block BERT (hidden=64), early stopping (patience=20, validation every 500 iterations).
**Rationale**: Retrieval needs rough ranking only — LLM ranker refines. Small BERT trains fast (500 epochs single GPU), reasonable initial ranking. Retrieval bottleneck modest; ranker compensates.
**Alternatives considered**: Larger BERT/Transformer (rejected: unnecessary for retrieval alone); BM25/tf-idf (rejected: needs text features for all items, ignores interaction history).

### TMDB + BLIP2 for multimodal features
**Decision**: Download TMDB posters, caption with BLIP2 (Salesforce/blip2-opt-2.7b).
**Rationale**: Movie recommendation benefits from visual features (genre cues from posters). BLIP2 gives textual descriptions usable by LLM ranker. TMDB standard movie metadata API, permissive access.
**Alternatives considered**: CLIP embeddings directly (rejected: less interpretable for LLM prompts); no visual features (rejected: weaker recommendations).

### Negative sampling in ranker training
**Decision**: Per positive candidate in ranker training, sample 19 negatives from retriever output (excluding gold item).
**Rationale**: Ranker must learn to distinguish correct item from plausible alternatives. 19 negatives = enough contrast without exploding batch size. Retriever output as negative pool → "hard" negatives (already ranked above most items).
**Alternatives considered**: Uniform random negatives (rejected: too easy, no fine-grained discrimination); all-pair ranking loss (rejected: O(n²), too slow).

## Risks / Trade-offs

- **[Risk] TMDB API dependency**: Full multimodal build needs `TMDB_API_KEY`; without it only text-only mode. → Text-only mode explicitly supported via `--skip-multimodal`, sufficient for smoke + baseline experiments.
- **[Risk] Cascading errors retriever→ranker**: Retriever misses gold item → ranker never recovers. → Retriever recall@K reported explicitly; retriever designed for high validation recall.
- **[Risk] Large dataset artifacts**: Full `dataset.pkl` with images + captions can be several GB. → Atomic writes prevent corruption; Parquet alternative enables selective loading.
- **[Risk] Qwen2.5 inference latency**: LLM ranker inference on all test candidates slow for large test sets. → Smoke scale limits test users; full scale acceptable for offline research evaluation.
