## Purpose

Evaluate trained recommendation models with standard ranking metrics on held-out test set.

## ADDED Requirements

### Requirement: Evaluate ranking metrics at top-K
System SHALL compute Hit Rate (HR), Recall, NDCG, MRR at configurable top-K (default: 1, 5, 10) on test split.

#### Scenario: Ranking evaluation on test set
- **WHEN** evaluation runs with `top_k=[1,5,10]` and `metrics=[hr, recall, ndcg, mrr]`
- **THEN** system outputs report with each metric at each top-K value

#### Scenario: Smoke-scale evaluation
- **WHEN** evaluation runs with `max_examples=null` (all test users) under `scale=smoke`
- **THEN** evaluation completes on reduced test set (32 users)

### Requirement: Use upstream evaluation interface
System SHALL support `use_upstream_eval=true` flag to delegate evaluation to upstream LLM evaluation framework.

#### Scenario: Upstream evaluation enabled
- **WHEN** `use_upstream_eval=true`
- **THEN** metrics computed by upstream evaluation module, not custom implementation
