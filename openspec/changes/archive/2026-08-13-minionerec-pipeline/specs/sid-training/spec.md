## Purpose

Train LoRA-adapted LLM to generate Semantic ID (SID) token sequences as item predictions — SFT warm-up followed by GRPO optimization with SID-specific rewards.

## ADDED Requirements

### Requirement: SFT stage uses SID-formatted prompts
System SHALL format training prompts with user-history items as SID tokens, target = SID sequence of held-out item.

#### Scenario: SID-prompt SFT training
- **WHEN** SID SFT loads data with `stages=[sft]`
- **THEN** each example formats history items as SID tokens (e.g., `<sid_12> <sid_5> <sid_47>`), target as next item's SID sequence

#### Scenario: Smoke-scale SID SFT
- **WHEN** SID SFT runs with `scale=smoke` (max_steps=4, batch_size=2)
- **THEN** system trains specified steps on SID-formatted data

### Requirement: GRPO stage generates SID sequences
System SHALL run GRPO where model generates SID token sequences, decoded to item IDs for reward computation.

#### Scenario: SID GRPO generation and decoding
- **WHEN** GRPO generates 3-token SID sequence
- **THEN** system decodes sequence to item ID, computes rewards from decoded item

#### Scenario: Handle invalid SID sequences
- **WHEN** GRPO generates sequence not matching any known SID code
- **THEN** system applies `invalid_penalty` (default: -0.5) to reward

### Requirement: SID-specific GRPO parameters
System SHALL support SID-specific GRPO generation params: `prefix_credit` for partial SID prefix matches, `max_completion_length=16` (for 3 SID tokens).

#### Scenario: Prefix credit rewards partial matches
- **WHEN** generated SID partially matches target (correct first token, wrong later)
- **THEN** system applies `prefix_credit` as partial reward
