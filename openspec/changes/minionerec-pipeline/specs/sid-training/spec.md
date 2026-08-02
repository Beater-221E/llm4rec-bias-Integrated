## Purpose

Train a LoRA-adapted LLM to generate Semantic ID (SID) token sequences as item predictions, using SFT warm-up followed by GRPO optimization with SID-specific rewards.

## ADDED Requirements

### Requirement: SFT stage uses SID-formatted prompts
The system SHALL format training prompts where items in the user history are represented as SID tokens and the target output is the SID sequence of the held-out item.

#### Scenario: SID-prompt SFT training
- **WHEN** SID SFT training loads data with `stages=[sft]`
- **THEN** each training example formats item history items as SID tokens (e.g., `<sid_12> <sid_5> <sid_47>`) and the target as the next item's SID sequence

#### Scenario: Smoke-scale SID SFT
- **WHEN** SID SFT runs with `scale=smoke` (max_steps=4, batch_size=2)
- **THEN** the system trains for the specified steps using SID-formatted data

### Requirement: GRPO stage generates SID sequences
The system SHALL run GRPO training where the model generates SID token sequences, which are decoded back to item IDs for reward computation.

#### Scenario: SID GRPO generation and decoding
- **WHEN** GRPO generates a 3-token SID sequence
- **THEN** the system decodes the SID sequence to the corresponding item ID and computes rewards based on the decoded item

#### Scenario: Handle invalid SID sequences
- **WHEN** GRPO generates a sequence that does not correspond to any known SID code
- **THEN** the system applies the `invalid_penalty` (default: -0.5) to the reward

### Requirement: SID-specific GRPO parameters
The system SHALL support SID-specific GRPO generation parameters including `prefix_credit` for partial SID prefix matches and `max_completion_length=16` (to accommodate 3 SID tokens).

#### Scenario: Prefix credit rewards partial matches
- **WHEN** a generated SID sequence partially matches the target (e.g., correct first token but wrong later tokens)
- **THEN** the system applies `prefix_credit` as a partial reward
