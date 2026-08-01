## Context

The mllm4rec pipeline operates independently from grpo4rec/minionerec. It uses its own codebase (`src/llm4rec_bias_Integrated/mllm4rec/`), its own CLI, and a different dataset (ml-latest-small instead of classic ML-100K). The architecture follows the two-stage paradigm common in sequential recommendation research: a fast retriever narrows the candidate space, then a more powerful ranker (LLM) re-ranks with full context.

Data preprocessing is a separate CLI (`build`) that handles TMDB API calls, image downloads, and BLIP2 captioning. Training has two separate CLIs for retriever and ranker.

## Goals / Non-Goals

**Goals:**
- Two-stage evaluation: the retriever must produce `retrieved.pkl` before ranker training
- Multimodal: item metadata includes text, poster images, and BLIP2 captions when available
- Text-only smoke mode: `--skip-multimodal` enables rapid testing without API dependencies

**Non-Goals:**
- Single-model recommendation — this track is explicitly two-stage
- Integrating with the grpo4rec/minionerec Hydra CLI — mllm4rec has its own config system
- End-to-end training (joint retriever+ranker optimization) — stages are trained independently

## Decisions

### Two-stage over single LLM
**Decision**: Retriever (BERT, 2 blocks, hidden=64) → Ranker (Qwen2.5-0.5B, LoRA).
**Rationale**: The candidate space (1,700 items) is too large for a single LLM to score efficiently in one pass. BERT retriever runs in O(n) over items and narrows to a manageable top-K. The LLM ranker then applies cross-attention with full context on a much smaller set. This mirrors established RecSys architectures (e.g., SASRec + BERT reranker).
**Alternatives considered**: Single LLM with listwise ranking (rejected: context window limits, VRAM); LLM as retriever (rejected: too slow for candidate generation over full corpus).

### BERT retriever over larger models
**Decision**: 2-block BERT (hidden=64) with early stopping (patience=20, validation every 500 iterations).
**Rationale**: Retrieval only needs a rough ranking — the LLM ranker will refine it. A small BERT is fast to train (500 epochs on single GPU) and produces a reasonable initial ranking. The retrieval quality bottleneck is modest because the ranker can compensate.
**Alternatives considered**: Larger BERT/Transformer (rejected: unnecessary for retrieval alone); BM25/tf-idf (rejected: needs text features for all items, doesn't use interaction history).

### TMDB + BLIP2 for multimodal features
**Decision**: Download TMDB posters, then caption with BLIP2 (Salesforce/blip2-opt-2.7b).
**Rationale**: Movie recommendation benefits from visual features (genre cues from posters). BLIP2 provides textual descriptions usable by the LLM ranker. TMDB is the standard movie metadata API with permissive access.
**Alternatives considered**: CLIP embeddings directly (rejected: less interpretable for LLM prompts); no visual features (rejected: weaker recommendations).

### Negative sampling in ranker training
**Decision**: For each positive candidate in ranker training, sample 19 negative candidates from the retriever's output (excluding the gold item).
**Rationale**: The ranker must learn to distinguish the correct item from plausible alternatives. 19 negatives provide enough contrast without exploding the batch size. Using retriever output as the negative pool ensures negatives are "hard" — the retriever already ranks them above most items.
**Alternatives considered**: Uniform random negatives (rejected: too easy, model doesn't learn fine-grained discrimination); all-pair ranking loss (rejected: O(n²) over candidates, too slow).

## Risks / Trade-offs

- **[Risk] TMDB API dependency**: Full multimodal build requires `TMDB_API_KEY`. Without it, only text-only mode works. → Text-only mode is explicitly supported with `--skip-multimodal` and is sufficient for smoke tests and baseline experiments.
- **[Risk] Cascading errors from retriever to ranker**: If the retriever misses the gold item, the ranker can never recover. → Recall@K of the retriever is reported explicitly; the retriever is designed to have high recall on the validation set.
- **[Risk] Large dataset artifacts**: Full `dataset.pkl` with images and captions can be several GB. → Atomic writes prevent corruption; Parquet alternative format enables selective loading.
- **[Risk] Qwen2.5 inference latency**: LLM ranker inference on all test candidates is slow for large test sets. → Smoke scale limits test users; full scale is acceptable for research evaluation (offline).
