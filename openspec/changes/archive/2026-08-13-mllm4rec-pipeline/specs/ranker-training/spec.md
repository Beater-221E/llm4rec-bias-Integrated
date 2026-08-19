## Purpose

Fine-tune Qwen2.5 with LoRA to re-rank candidate items retrieved by retriever, using negative sampling + multi-modal item information.

## ADDED Requirements

### Requirement: Load retriever candidates
System SHALL load candidate items from `retrieved.pkl` produced by retriever stage.

#### Scenario: Candidate loading
- **WHEN** ranker training starts with `retrieved_pkl=experiments/lru/ml-100k/retrieved.pkl`
- **THEN** system loads top-K candidate list per user for re-ranking

### Requirement: LoRA fine-tune Qwen2.5 with negative sampling
System SHALL train LoRA-adapted Qwen2.5-0.5B-Instruct to rank candidates using negative sampling (19 negatives per positive).

#### Scenario: Ranker LoRA training
- **WHEN** ranker training runs with default config (lora_num_epochs=3, micro_batch_size=2, lr=1e-4, negative_sample_size=19, max_history=25)
- **THEN** system trains model to rank candidates by scoring each against user history

#### Scenario: Smoke-scale ranker
- **WHEN** ranker training runs with `scale=smoke` (lora_num_epochs=1, max_train_steps=20)
- **THEN** system trains at most 20 steps or 1 epoch, whichever first

### Requirement: Include item metadata in prompts
System SHALL format ranker prompts with user history items incl. available metadata (title, poster, caption), truncated to `max_history=25` items.

#### Scenario: Metadata-rich prompts
- **WHEN** ranker formats training prompt and multimodal data available
- **THEN** each history item includes title + (if available) image + caption references
