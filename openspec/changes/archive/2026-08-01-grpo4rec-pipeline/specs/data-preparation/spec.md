## Purpose

Provide a reproducible pipeline to download, preprocess, and split the classic MovieLens-100K dataset for LLM-based recommendation training and evaluation.

## ADDED Requirements

### Requirement: Download MovieLens-100K raw data
The system SHALL download the classic MovieLens-100K dataset (`u.data`, `u.item`) and store it in `data/raw/movielens_100k/`.

#### Scenario: First-time download
- **WHEN** the user runs `python -m llm4rec.cli.main prepare dataset=movielens_100k` and no raw data exists
- **THEN** the system downloads the dataset files and saves them under `data/raw/movielens_100k/`

#### Scenario: Idempotent re-run
- **WHEN** the user runs the same prepare command and raw data already exists
- **THEN** the system skips download and proceeds to preprocessing

### Requirement: Preprocess interactions into train/val/test splits
The system SHALL process raw ratings into user interaction sequences, split with leave-one-out strategy (last item as test, second-to-last as val), and persist to `data/processed/movielens_100k/`.

#### Scenario: Standard preprocessing
- **WHEN** raw data is present and preprocessing runs with default config (rating_threshold=4.0, min_user_interactions=5, history_max_length=20)
- **THEN** the system writes serialized train/val/test split files to `data/processed/movielens_100k/` with user histories and target items

#### Scenario: Smoke-scale preprocessing
- **WHEN** preprocessing runs with `scale=smoke` (train_limit=64, eval_limit=32, history_max_length=8)
- **THEN** the system produces a smaller dataset suitable for rapid testing

### Requirement: Compute item popularity and framing metadata
The system SHALL compute per-item popularity statistics and framing metadata during preprocessing for use in bias evaluation.

#### Scenario: Popularity computation
- **WHEN** preprocessing completes
- **THEN** each item in the processed dataset has an associated popularity score derived from interaction counts
