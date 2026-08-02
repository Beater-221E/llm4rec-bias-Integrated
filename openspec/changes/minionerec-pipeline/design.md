## Context

The minionerec pipeline builds directly on grpo4rec's infrastructure. It reuses the same MovieLens-100K processed data, the same `src/llm4rec/` codebase, the same Hydra CLI, and the same training loop. The key addition is the Semantic ID (SID) layer: instead of predicting raw item text, the model predicts 3-token hierarchical codes.

Existing `config.yaml` has the full minionerec section with SID-specific parameters (method, levels, codebook_size, collision_handling) and SID-aware training stages (SFT, GRPO, evaluation).

## Goals / Non-Goals

**Goals:**
- Reuse grpo4rec's data preprocessing — only generate SID artifacts on top of existing processed data
- SID generation is idempotent: re-running `prepare` with the same config produces identical codebook and mappings
- Training auto-builds SID if missing (convenience for new runs)

**Non-Goals:**
- Changing the underlying recommendation model architecture — same Qwen2.5 + LoRA setup
- Supporting non-residual SID methods — only residual K-means is implemented
- SID transfer across datasets — codebook is dataset-specific

## Decisions

### Residual K-means for SID construction
**Decision**: Use 3-level residual K-means with codebook size 64 per level.
**Rationale**: Residual clustering decomposes the item embedding space hierarchically. Each level clusters the residual from the previous level, so the 3-token SID refines from coarse to fine. 64 codes per level gives 64³ ≈ 262K possible items, far exceeding the ~1,700 items in ML-100K.
**Alternatives considered**: Flat K-means (rejected: loses hierarchical structure); product quantization (rejected: more complex, no clear benefit for this vocabulary size); RQ-VAE (rejected: requires training a VAE, adds complexity).

### Extra level collision handling
**Decision**: When two items share the same 3-level SID, add additional clustering levels until codes are unique.
**Rationale**: With 64³ possible codes and ~1,700 items, collisions are rare but possible. Extra levels are a simple deterministic resolution that preserves the first 3 tokens (model can still generate partial matches).
**Alternatives considered**: Reject/ignore duplicates (rejected: loses items); re-hash (rejected: non-deterministic).

### Prefix credit in GRPO rewards
**Decision**: Award partial reward (`prefix_credit=0.1`) when generated SID partially matches the target (e.g., first token correct).
**Rationale**: Hierarchical SIDs mean early tokens constrain the search space more than later tokens. Prefix credit rewards the model for getting the coarse category right, providing a smoother reward signal.
**Alternatives considered**: Binary reward only (rejected: sparse signal, harder to learn); full edit distance (rejected: too complex for 3 tokens).

## Risks / Trade-offs

- **[Risk] SID collision degrades evaluation**: If the codebook has collisions that extra levels don't fully resolve, two items with the same SID are indistinguishable. → Collision rate is reported in evaluation; codebook_size=64 with 3 levels should minimize this for ~1,700 items.
- **[Risk] Residual clustering quality depends on item embeddings**: If embeddings are poor (e.g., from a weak pretrained model), cluster structure is noisy. → Currently uses embedding quality tied to the dataset; future work could use better embeddings.
- **[Risk] SID vocabulary size impacts generation**: Model must learn 64×3=192 tokens on top of the base vocabulary. With Qwen2.5-0.5B, this is manageable but adds generation complexity vs. raw text prediction.
