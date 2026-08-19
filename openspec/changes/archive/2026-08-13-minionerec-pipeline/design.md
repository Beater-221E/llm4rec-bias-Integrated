## Context

minionerec builds on grpo4rec infrastructure. Reuses same MovieLens-100K processed data, same `src/llm4rec/` codebase, same Hydra CLI, same training loop. Key addition: Semantic ID (SID) layer — model predicts 3-token hierarchical codes instead of raw item text.

Existing `config.yaml` has full minionerec section: SID params (method, levels, codebook_size, collision_handling) + SID-aware stages (SFT, GRPO, evaluation).

## Goals / Non-Goals

**Goals:**
- Reuse grpo4rec preprocessing — only generate SID artifacts on top of existing processed data
- SID generation idempotent: re-running `prepare` with same config → identical codebook + mappings
- Training auto-builds SID if missing (convenience for new runs)

**Non-Goals:**
- Change underlying recommendation model — same Qwen2.5 + LoRA
- Non-residual SID methods — residual K-means only
- SID transfer across datasets — codebook dataset-specific

## Decisions

### Residual K-means for SID construction
**Decision**: 3-level residual K-means, codebook size 64 per level.
**Rationale**: Residual clustering decomposes item embedding space hierarchically. Each level clusters previous level's residual → 3-token SID refines coarse→fine. 64³ ≈ 262K possible items, far exceeds ~1,700 ML-100K items.
**Alternatives considered**: Flat K-means (rejected: loses hierarchy); product quantization (rejected: more complex, no benefit at this vocab size); RQ-VAE (rejected: needs VAE training, adds complexity).

### Extra level collision handling
**Decision**: Two items sharing same 3-level SID → add clustering levels until codes unique.
**Rationale**: 64³ codes vs ~1,700 items → collisions rare but possible. Extra levels = simple deterministic resolution, preserves first 3 tokens (partial-match generation still works).
**Alternatives considered**: Reject/ignore duplicates (rejected: loses items); re-hash (rejected: non-deterministic).

### Prefix credit in GRPO rewards
**Decision**: Partial reward (`prefix_credit=0.1`) when generated SID partially matches target (e.g., first token correct).
**Rationale**: Hierarchical SIDs → early tokens constrain search more than later. Prefix credit rewards coarse-category correctness, smoother reward signal.
**Alternatives considered**: Binary reward only (rejected: sparse, harder to learn); full edit distance (rejected: too complex for 3 tokens).

## Risks / Trade-offs

- **[Risk] SID collision degrades evaluation**: unresolved collisions make two items indistinguishable. → Collision rate reported in evaluation; codebook_size=64 × 3 levels should minimize for ~1,700 items.
- **[Risk] Residual clustering quality depends on item embeddings**: poor embeddings → noisy cluster structure. → Current embeddings tied to dataset; future work could use better ones.
- **[Risk] SID vocabulary size impacts generation**: model must learn 64×3=192 extra tokens. With Qwen2.5-0.5B manageable, but adds generation complexity vs. raw text prediction.
