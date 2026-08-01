## Purpose

Evaluate the entire two-stage recommendation pipeline end-to-end, measuring retrieval recall and final ranking quality on the test split.

## ADDED Requirements

### Requirement: Evaluate retriever recall
The system SHALL compute recall metrics for the retriever stage, measuring how often the target item appears in the retriever's top-K candidates.

#### Scenario: Retriever recall evaluation
- **WHEN** evaluation runs on the retriever output
- **THEN** the system reports recall@K metrics for the retriever's candidate generation

### Requirement: Evaluate ranker ranking quality
The system SHALL compute ranking metrics (NDCG, MRR, Hit Rate) on the output of the ranker stage.

#### Scenario: Ranker ranking evaluation
- **WHEN** the ranker re-ranks candidates for test users
- **THEN** the system reports final ranking metrics on the re-ranked output

### Requirement: End-to-end pipeline evaluation
The system SHALL support evaluating the full retriever → ranker pipeline, reporting both retrieval and ranking metrics in a single run.

#### Scenario: Full pipeline evaluation
- **WHEN** evaluation runs with both retriever and ranker outputs available
- **THEN** the system reports a combined evaluation showing recall from the retriever and final ranking metrics from the ranker
