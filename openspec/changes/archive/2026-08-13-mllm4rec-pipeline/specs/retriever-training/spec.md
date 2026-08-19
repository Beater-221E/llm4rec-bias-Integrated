## Purpose

Train lightweight BERT-based retriever (SASRec-style) to produce ranked candidate set for downstream LLM re-ranking.

## ADDED Requirements

### Requirement: Load data from dataset.pkl
System SHALL load preprocessed interaction data from `dataset.pkl` specified in config.

#### Scenario: Dataset loading
- **WHEN** retriever training starts with `dataset_pkl=data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/dataset.pkl`
- **THEN** system loads user sequences, item metadata, split assignments from pickle

### Requirement: Train SASRec-style BERT retriever
System SHALL train configurable BERT-based retriever (2 blocks, hidden_dim=64, max_len=200, dropout=0.2) with early stopping on validation.

#### Scenario: Full retriever training
- **WHEN** retriever training runs with default config (num_epochs=500, early_stopping_patience=20, val_strategy=iteration, val_iterations=500)
- **THEN** system trains up to 500 epochs, evaluates on validation every 500 iterations, stops early if no improvement for 20 evaluations

#### Scenario: Smoke-scale retriever
- **WHEN** retriever training runs with `scale=smoke` (num_epochs=2)
- **THEN** system trains exactly 2 epochs regardless of early stopping

### Requirement: Export retrieved candidates
System SHALL produce `retrieved.pkl` with retriever's top-K ranked candidates per user for ranker.

#### Scenario: Candidate export
- **WHEN** retriever training completes
- **THEN** system writes `retrieved.pkl` to `experiments/lru/ml-100k/retrieved.pkl`
