## Purpose

Evaluate SID-based recommendation models using both ranking metrics and SID-specific generation quality metrics including validity and collision rates.

## ADDED Requirements

### Requirement: Evaluate ranking metrics with SID decoding
The system SHALL compute ranking metrics (at top-K = 1, 5, 10) by decoding generated SID sequences to item predictions for test users.

#### Scenario: SID ranking evaluation
- **WHEN** evaluation runs on the SID model with `top_k=[1,5,10]`
- **THEN** the system generates SID sequences for test prompts, decodes them to items, and computes ranking metrics

### Requirement: Free-generation evaluation
The system SHALL support free-generation evaluation where the model generates `free_gen_n` items without constraining to a candidate set.

#### Scenario: Free-generation evaluation
- **WHEN** evaluation runs with `free_gen_n=50`
- **THEN** the model generates up to 50 items per user and evaluation captures the full generation quality

#### Scenario: Smoke-scale free generation
- **WHEN** evaluation runs with `scale=smoke` and `free_gen_n=4`
- **THEN** the model generates up to 4 items per user for the reduced test set

### Requirement: Measure SID validity
The system SHALL report the proportion of generated SID sequences that decode to valid items (SID validity rate).

#### Scenario: SID validity measurement
- **WHEN** evaluation completes
- **THEN** the report includes a `sid_validity` metric showing the fraction of generated sequences that successfully decode

### Requirement: Measure semantic collision rate
The system SHALL report the rate at which distinct generated SID sequences decode to the same item (collision rate).

#### Scenario: Collision rate measurement
- **WHEN** multiple generated sequences decode to the same item
- **THEN** the report includes a `semantic_collision` metric capturing this behavior

### Requirement: Reuse upstream evaluation
The system SHALL support `use_upstream_eval=true` to delegate ranking metric computation to the upstream framework, with SID-specific metrics added on top.

#### Scenario: Upstream evaluation with SID add-ons
- **WHEN** evaluation runs with `use_upstream_eval=true`
- **THEN** ranking metrics come from upstream evaluation while SID validity and collision metrics are computed by the SID evaluation module
