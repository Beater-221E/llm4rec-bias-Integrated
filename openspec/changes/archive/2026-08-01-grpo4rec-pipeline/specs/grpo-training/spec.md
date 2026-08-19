## Purpose

SFT followed by Group Relative Policy Optimization (GRPO) training of LoRA-adapted Qwen2.5 on candidate-choice recommendation task.

## ADDED Requirements

### Requirement: SFT stage teaches the task format
System SHALL run supervised fine-tuning stage training LoRA-adapted model on formatted recommendation prompts where target output = correct item identifier.

#### Scenario: SFT completes with smoke scale
- **WHEN** user runs SFT with `scale=smoke` (max_steps=4, batch_size=1, lr=1e-4, rank=8)
- **THEN** system trains exactly 4 steps, logs progress every step, saves checkpoint

#### Scenario: SFT uses leave-one-out splits
- **WHEN** SFT loads data
- **THEN** uses train split; each user's history (excluding last item) = context, last item = target

### Requirement: GRPO stage optimizes candidate selection
System SHALL run GRPO generating multiple candidate predictions per prompt, scoring with reward components, updating policy.

#### Scenario: GRPO generates and scores candidates
- **WHEN** GRPO runs with default reward weights (exact_match=1.0, format_validity=0.2)
- **THEN** system generates `num_generations` candidates per prompt, computes reward per candidate, applies GRPO policy updates

#### Scenario: GRPO handles invalid outputs
- **WHEN** generated output fails expected format
- **THEN** system applies `invalid_penalty` (default -0.5) to discourage invalid generations

### Requirement: LoRA adapters are configurable
System SHALL support configurable LoRA params (rank, alpha, dropout, target modules) for SFT + GRPO stages.

#### Scenario: LoRA configuration from config
- **WHEN** training starts with `peft.method=lora, peft.rank=16, peft.alpha=32, peft.target_modules=[q_proj, k_proj, v_proj, o_proj]`
- **THEN** model loads with specified LoRA config applied

### Requirement: Resume or reuse preprocessed data
System SHALL reuse existing preprocessed data when available instead of re-running preprocessing.

#### Scenario: Train after prior prepare
- **WHEN** `train` invoked and `data/processed/movielens_100k/` exists
- **THEN** system loads existing processed data, skips preprocessing
