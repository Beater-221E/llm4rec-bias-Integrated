## Purpose

Evaluate SID-based recommendation models with ranking metrics + SID-specific generation quality metrics (validity, collision rates).

## ADDED Requirements

### Requirement: Evaluate ranking metrics with SID decoding
System SHALL compute ranking metrics (top-K = 1, 5, 10) by decoding generated SID sequences to item predictions for test users.

#### Scenario: SID ranking evaluation
- **WHEN** evaluation runs on SID model with `top_k=[1,5,10]`
- **THEN** system generates SID sequences for test prompts, decodes to items, computes ranking metrics

### Requirement: Free-generation evaluation
System SHALL support free-generation evaluation where model generates `free_gen_n` items without candidate-set constraint.

#### Scenario: Free-generation evaluation
- **WHEN** evaluation runs with `free_gen_n=50`
- **THEN** model generates up to 50 items per user, evaluation captures full generation quality

#### Scenario: Smoke-scale free generation
- **WHEN** evaluation runs with `scale=smoke` and `free_gen_n=4`
- **THEN** model generates up to 4 items per user for reduced test set

### Requirement: Measure SID validity
System SHALL report proportion of generated SID sequences decoding to valid items (SID validity rate).

#### Scenario: SID validity measurement
- **WHEN** evaluation completes
- **THEN** report includes `sid_validity` metric showing fraction of sequences successfully decoding

### Requirement: Measure semantic collision rate
System SHALL report rate at which distinct generated SID sequences decode to same item (collision rate).

#### Scenario: Collision rate measurement
- **WHEN** multiple generated sequences decode to same item
- **THEN** report includes `semantic_collision` metric capturing behavior

### Requirement: Reuse upstream evaluation
System SHALL support `use_upstream_eval=true` to delegate ranking metric computation to upstream framework, with SID-specific metrics added on top.

#### Scenario: Upstream evaluation with SID add-ons
- **WHEN** evaluation runs with `use_upstream_eval=true`
- **THEN** ranking metrics from upstream evaluation; SID validity + collision metrics from SID evaluation module
