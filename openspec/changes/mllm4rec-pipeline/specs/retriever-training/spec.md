## Purpose

Train a lightweight BERT-based retriever model (SASRec-style) to produce a ranked candidate set for downstream LLM re-ranking.

## ADDED Requirements

### Requirement: Load data from dataset.pkl
The system SHALL load preprocessed interaction data from the `dataset.pkl` file specified in config.

#### Scenario: Dataset loading
- **WHEN** retriever training starts with `dataset_pkl=data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/dataset.pkl`
- **THEN** the system loads user sequences, item metadata, and split assignments from the pickle file

### Requirement: Train SASRec-style BERT retriever
The system SHALL train a configurable BERT-based retriever (2 blocks, hidden_dim=64, max_len=200, dropout=0.2) with early stopping on validation.

#### Scenario: Full retriever training
- **WHEN** retriever training runs with default config (num_epochs=500, early_stopping_patience=20, val_strategy=iteration, val_iterations=500)
- **THEN** the system trains for up to 500 epochs, evaluating on the validation set every 500 iterations, stopping early if no improvement for 20 evaluations

#### Scenario: Smoke-scale retriever
- **WHEN** retriever training runs with `scale=smoke` (num_epochs=2)
- **THEN** the system trains for exactly 2 epochs regardless of early stopping

### Requirement: Export retrieved candidates
The system SHALL produce a `retrieved.pkl` file containing the retriever's top-K ranked candidates per user for the ranker to consume.

#### Scenario: Candidate export
- **WHEN** retriever training completes
- **THEN** the system writes `retrieved.pkl` to `experiments/lru/ml-100k/retrieved.pkl`
