## Purpose

Fine-tune a Qwen2.5 LLM with LoRA to re-rank candidate items retrieved by the retriever, using negative sampling and multi-modal item information.

## ADDED Requirements

### Requirement: Load retriever candidates
The system SHALL load candidate items from the `retrieved.pkl` file produced by the retriever stage.

#### Scenario: Candidate loading
- **WHEN** ranker training starts with `retrieved_pkl=experiments/lru/ml-100k/retrieved.pkl`
- **THEN** the system loads the top-K candidate list per user for re-ranking

### Requirement: LoRA fine-tune Qwen2.5 with negative sampling
The system SHALL train a LoRA-adapted Qwen2.5-0.5B-Instruct model to rank candidates using negative sampling (19 negatives per positive).

#### Scenario: Ranker LoRA training
- **WHEN** ranker training runs with default config (lora_num_epochs=3, micro_batch_size=2, lr=1e-4, negative_sample_size=19, max_history=25)
- **THEN** the system trains the model to rank candidates by scoring each candidate against the user history

#### Scenario: Smoke-scale ranker
- **WHEN** ranker training runs with `scale=smoke` (lora_num_epochs=1, max_train_steps=20)
- **THEN** the system trains for at most 20 steps or 1 epoch, whichever comes first

### Requirement: Include item metadata in prompts
The system SHALL format ranker prompts with user history items including available metadata (title, poster, caption) truncated to `max_history=25` items.

#### Scenario: Metadata-rich prompts
- **WHEN** ranker formats a training prompt and multimodal data is available
- **THEN** each history item includes its title and, if available, image and caption references
