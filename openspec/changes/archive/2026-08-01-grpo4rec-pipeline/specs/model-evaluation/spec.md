## Purpose

Evaluate trained recommendation models using standard ranking metrics on a held-out test set to measure recommendation quality.

## ADDED Requirements

### Requirement: Evaluate ranking metrics at top-K
The system SHALL compute Hit Rate (HR), Recall, NDCG, and MRR at configurable top-K values (default: 1, 5, 10) on the test split.

#### Scenario: Ranking evaluation on test set
- **WHEN** evaluation runs with `top_k=[1,5,10]` and `metrics=[hr, recall, ndcg, mrr]`
- **THEN** the system outputs a report with each metric at each top-K value

#### Scenario: Smoke-scale evaluation
- **WHEN** evaluation runs with `max_examples=null` (use all test users) under `scale=smoke`
- **THEN** evaluation completes on the reduced test set (32 users)

### Requirement: Use upstream evaluation interface
The system SHALL support the `use_upstream_eval=true` flag to delegate evaluation to the upstream LLM evaluation framework.

#### Scenario: Upstream evaluation enabled
- **WHEN** `use_upstream_eval=true`
- **THEN** evaluation metrics are computed by the upstream evaluation module rather than a custom implementation
