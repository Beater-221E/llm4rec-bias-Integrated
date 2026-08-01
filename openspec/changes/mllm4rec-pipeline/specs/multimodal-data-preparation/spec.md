## Purpose

Preprocess the ml-latest-small (and optionally ml-1m) dataset into train/val/test interaction splits with item metadata, poster images, and BLIP2 captions for multi-modal recommendation.

## ADDED Requirements

### Requirement: Build interaction dataset with metadata
The system SHALL process raw ml-latest-small (or ml-1m) data into interaction sequences, apply iterative k-core filtering (min_uc=5, min_sc=5), and serialize to `dataset.pkl` and Parquet.

#### Scenario: Full interaction build
- **WHEN** data build runs with default config (min_rating=0, min_uc=5, min_sc=5, iterative_kcore=true)
- **THEN** the system writes `dataset.pkl` and Parquet files to `data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/`

#### Scenario: Text-only mode (smoke)
- **WHEN** data build runs with `--skip-multimodal` flag
- **THEN** the system builds `dataset.pkl` with only interaction data and item titles, without fetching posters or generating captions

### Requirement: Fetch TMDB posters
The system SHALL download movie poster images from TMDB using the configured API key and store them alongside the dataset.

#### Scenario: Poster download
- **WHEN** data build runs without `--skip-multimodal` and `TMDB_API_KEY` is set
- **THEN** the system downloads poster images to the output directory with resume support for interrupted downloads

#### Scenario: TMDB key missing
- **WHEN** data build runs without `--skip-multimodal` and `TMDB_API_KEY` is not set
- **THEN** the system reports an error indicating the missing API key

### Requirement: Generate BLIP2 captions
The system SHALL generate image captions for each movie poster using BLIP2 (Salesforce/blip2-opt-2.7b) on CUDA.

#### Scenario: Caption generation
- **WHEN** posters are downloaded and BLIP2 is available on CUDA
- **THEN** the system generates per-movie captions with configurable batch_size and resume support

### Requirement: Atomic writes with backup
The system SHALL use atomic writes and create backups during dataset serialization to prevent corruption.

#### Scenario: Atomic serialization
- **WHEN** dataset is written to disk
- **THEN** the system writes to a temp file first, then atomically renames, with `create_backup=true` preserving the previous version
