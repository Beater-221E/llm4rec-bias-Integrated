## Purpose

Preprocess ml-latest-small (optionally ml-1m) into train/val/test interaction splits with item metadata, poster images, BLIP2 captions for multi-modal recommendation.

## ADDED Requirements

### Requirement: Build interaction dataset with metadata
System SHALL process raw ml-latest-small (or ml-1m) into interaction sequences, iterative k-core filtering (min_uc=5, min_sc=5), serialize to `dataset.pkl` + Parquet.

#### Scenario: Full interaction build
- **WHEN** data build runs with default config (min_rating=0, min_uc=5, min_sc=5, iterative_kcore=true)
- **THEN** system writes `dataset.pkl` + Parquet files to `data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/`

#### Scenario: Text-only mode (smoke)
- **WHEN** data build runs with `--skip-multimodal` flag
- **THEN** system builds `dataset.pkl` with only interaction data + item titles, without posters or captions

### Requirement: Fetch TMDB posters
System SHALL download movie poster images from TMDB using configured API key, store alongside dataset.

#### Scenario: Poster download
- **WHEN** data build runs without `--skip-multimodal` and `TMDB_API_KEY` set
- **THEN** system downloads posters to output directory with resume support for interrupted downloads

#### Scenario: TMDB key missing
- **WHEN** data build runs without `--skip-multimodal` and `TMDB_API_KEY` not set
- **THEN** system reports error indicating missing API key

### Requirement: Generate BLIP2 captions
System SHALL generate image captions per movie poster using BLIP2 (Salesforce/blip2-opt-2.7b) on CUDA.

#### Scenario: Caption generation
- **WHEN** posters downloaded and BLIP2 available on CUDA
- **THEN** system generates per-movie captions with configurable batch_size + resume support

### Requirement: Atomic writes with backup
System SHALL use atomic writes + backups during dataset serialization to prevent corruption.

#### Scenario: Atomic serialization
- **WHEN** dataset written to disk
- **THEN** system writes temp file first, atomically renames, `create_backup=true` preserves previous version
