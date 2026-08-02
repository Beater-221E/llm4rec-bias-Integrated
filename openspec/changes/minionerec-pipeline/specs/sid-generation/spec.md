## Purpose

Build hierarchical Semantic ID (SID) codes for items using residual K-means clustering, enabling compact and structured item representation for sequence generation.

## ADDED Requirements

### Requirement: Build residual K-means codebook
The system SHALL construct a 3-level residual K-means codebook with configurable codebook size (default: 64) from item embeddings derived from the MovieLens-100K processed data.

#### Scenario: Codebook construction
- **WHEN** SID preparation runs with `method=residual_kmeans`, `levels=3`, `codebook_size=64`
- **THEN** the system produces a codebook file at `data/processed/movielens_100k/sid/codebook.json`

#### Scenario: Smoke-scale codebook
- **WHEN** SID preparation runs under `scale=smoke` with `codebook_size=64`
- **THEN** a codebook is built from the reduced smoke dataset and written to the same location

### Requirement: Map items to SID sequences
The system SHALL generate a 3-token SID sequence for every item and persist the mapping as JSONL files for train, val, and test splits.

#### Scenario: SID mapping generation
- **WHEN** the codebook is built and items are clustered
- **THEN** the system writes `sid_train.jsonl`, `sid_val.jsonl`, and `sid_test.jsonl` mapping item IDs to their 3-token SID sequences

### Requirement: Handle SID collisions
The system SHALL handle cases where two items map to the same SID sequence using the configured `collision_handling` strategy (default: `extra_level`).

#### Scenario: Collision resolution
- **WHEN** two items map to identical SID codes after 3 levels
- **THEN** the system applies the `extra_level` collision strategy, adding additional clustering levels until codes are unique

### Requirement: Reuse preprocessed data
The system SHALL build SID codes on top of existing `data/processed/movielens_100k/` data without re-running the full preprocessing pipeline.

#### Scenario: SID on existing processed data
- **WHEN** `data/processed/movielens_100k/` exists from a prior grpo4rec prepare run
- **THEN** SID preparation reads from the existing processed splits and only generates SID artifacts
