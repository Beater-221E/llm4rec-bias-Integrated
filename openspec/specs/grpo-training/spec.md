# grpo-training Specification

## Purpose

Support supervised fine-tuning (SFT) followed by Group Relative Policy Optimization (GRPO) training of a LoRA-adapted Qwen2.5 model on the candidate-choice recommendation task.

## Requirements

### Requirement: SFT stage teaches the task format
The system SHALL run a supervised fine-tuning stage that trains the LoRA-adapted model on formatted recommendation prompts where the target output is the correct item identifier.

#### Scenario: SFT completes with smoke scale
- **WHEN** the user runs SFT training with `scale=smoke` (max_steps=4, batch_size=1, lr=1e-4, rank=8)
- **THEN** the system trains for exactly 4 steps, logs progress every step, and saves a checkpoint

#### Scenario: SFT uses leave-one-out splits
- **WHEN** SFT training loads data
- **THEN** it uses the train split to build prompts where each user's history (excluding the last item) forms the context and the last item is the target

### Requirement: GRPO stage optimizes candidate selection
The system SHALL run GRPO training that generates multiple candidate predictions per prompt, scores them with reward components, and updates the policy.

#### Scenario: GRPO generates and scores candidates
- **WHEN** GRPO training runs with default reward weights (exact_match=1.0, format_validity=0.2)
- **THEN** the system generates `num_generations` candidates per prompt, computes a reward for each, and applies GRPO policy updates

#### Scenario: GRPO handles invalid outputs
- **WHEN** a generated output does not match the expected format
- **THEN** the system applies an `invalid_penalty` (default -0.5) to discourage invalid generations

### Requirement: LoRA adapters are configurable
The system SHALL support configurable LoRA parameters (rank, alpha, dropout, target modules) for both SFT and GRPO stages.

#### Scenario: LoRA configuration from config
- **WHEN** training starts with `peft.method=lora, peft.rank=16, peft.alpha=32, peft.target_modules=[q_proj, k_proj, v_proj, o_proj]`
- **THEN** the model loads with the specified LoRA configuration applied

### Requirement: Resume or reuse preprocessed data
The system SHALL reuse existing preprocessed data when available instead of re-running preprocessing.

#### Scenario: Train after prior prepare
- **WHEN** `train` is invoked and `data/processed/movielens_100k/` already exists
- **THEN** the system loads the existing processed data and skips preprocessing
