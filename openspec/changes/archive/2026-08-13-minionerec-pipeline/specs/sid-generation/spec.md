## Purpose

Build hierarchical Semantic ID (SID) codes for items via residual K-means clustering — compact, structured item representation for sequence generation.

## ADDED Requirements

### Requirement: Build residual K-means codebook
System SHALL construct 3-level residual K-means codebook with configurable size (default: 64) from item embeddings derived from MovieLens-100K processed data.

#### Scenario: Codebook construction
- **WHEN** SID preparation runs with `method=residual_kmeans`, `levels=3`, `codebook_size=64`
- **THEN** system produces codebook at `data/processed/movielens_100k/sid/codebook.json`

#### Scenario: Smoke-scale codebook
- **WHEN** SID preparation runs under `scale=smoke` with `codebook_size=64`
- **THEN** codebook built from reduced smoke dataset, written to same location

### Requirement: Map items to SID sequences
System SHALL generate 3-token SID sequence per item, persist mapping as JSONL for train/val/test splits.

#### Scenario: SID mapping generation
- **WHEN** codebook built and items clustered
- **THEN** system writes `sid_train.jsonl`, `sid_val.jsonl`, `sid_test.jsonl` mapping item IDs to 3-token SID sequences

### Requirement: Handle SID collisions
System SHALL handle two items mapping to same SID sequence via configured `collision_handling` strategy (default: `extra_level`).

#### Scenario: Collision resolution
- **WHEN** two items map to identical SID codes after 3 levels
- **THEN** system applies `extra_level` strategy, adding clustering levels until codes unique

### Requirement: Reuse preprocessed data
System SHALL build SID codes on top of existing `data/processed/movielens_100k/` without re-running full preprocessing.

#### Scenario: SID on existing processed data
- **WHEN** `data/processed/movielens_100k/` exists from prior grpo4rec prepare run
- **THEN** SID preparation reads existing processed splits, only generates SID artifacts
