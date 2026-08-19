# data-preparation Specification

## Purpose

Reproducible pipeline to download, preprocess, split classic MovieLens-100K for LLM-based recommendation training + evaluation.

## Requirements

### Requirement: Download MovieLens-100K raw data
System SHALL download classic MovieLens-100K (`u.data`, `u.item`), store in `data/raw/movielens_100k/`.

#### Scenario: First-time download
- **WHEN** user runs `python -m llm4rec.cli.main prepare dataset=movielens_100k` and no raw data exists
- **THEN** system downloads files to `data/raw/movielens_100k/`

#### Scenario: Idempotent re-run
- **WHEN** user runs same prepare command and raw data exists
- **THEN** system skips download, proceeds to preprocessing

### Requirement: Preprocess interactions into train/val/test splits
System SHALL process raw ratings into user interaction sequences, leave-one-out split (last item test, second-to-last val), persist to `data/processed/movielens_100k/`.

#### Scenario: Standard preprocessing
- **WHEN** raw data present, preprocessing runs with default config (rating_threshold=4.0, min_user_interactions=5, history_max_length=20)
- **THEN** system writes serialized train/val/test split files to `data/processed/movielens_100k/` with user histories + target items

#### Scenario: Smoke-scale preprocessing
- **WHEN** preprocessing runs with `scale=smoke` (train_limit=64, eval_limit=32, history_max_length=8)
- **THEN** system produces smaller dataset for rapid testing

### Requirement: Compute item popularity and framing metadata
System SHALL compute per-item popularity stats + framing metadata during preprocessing for bias evaluation.

#### Scenario: Popularity computation
- **WHEN** preprocessing completes
- **THEN** each processed item has popularity score derived from interaction counts
