## Purpose

Evaluate entire two-stage pipeline end-to-end: retrieval recall + final ranking quality on test split.

## ADDED Requirements

### Requirement: Evaluate retriever recall
System SHALL compute recall metrics for retriever stage, measuring how often target item appears in retriever's top-K candidates.

#### Scenario: Retriever recall evaluation
- **WHEN** evaluation runs on retriever output
- **THEN** system reports recall@K metrics for retriever candidate generation

### Requirement: Evaluate ranker ranking quality
System SHALL compute ranking metrics (NDCG, MRR, Hit Rate) on ranker stage output.

#### Scenario: Ranker ranking evaluation
- **WHEN** ranker re-ranks candidates for test users
- **THEN** system reports final ranking metrics on re-ranked output

### Requirement: End-to-end pipeline evaluation
System SHALL support evaluating full retriever → ranker pipeline, reporting retrieval + ranking metrics in single run.

#### Scenario: Full pipeline evaluation
- **WHEN** evaluation runs with both retriever + ranker outputs available
- **THEN** system reports combined evaluation: retriever recall + ranker final ranking metrics
