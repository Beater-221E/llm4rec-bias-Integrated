# bias-analysis Specification

## Purpose

Measure recommendation bias across multiple dimensions (popularity, position, framing, recency) to understand and diagnose fairness issues in the trained model.

## Requirements

### Requirement: Probe popularity bias
The system SHALL measure how model recommendations correlate with item popularity, reporting a popularity-bias score.

#### Scenario: Popularity bias analysis
- **WHEN** bias analysis runs with `probes=[popularity]`
- **THEN** the system computes a popularity bias metric comparing recommended items against the overall item popularity distribution

### Requirement: Probe position bias
The system SHALL measure whether the model is biased by the position of items in the input sequence.

#### Scenario: Position bias analysis
- **WHEN** bias analysis runs with `probes=[position]`
- **THEN** the system reports a position-bias score indicating whether earlier or later items in the history are favored

### Requirement: Probe framing bias
The system SHALL measure whether the model's recommendations change based on how prompts are framed (e.g., neutral vs. biased prompt phrasing).

#### Scenario: Framing bias analysis
- **WHEN** bias analysis runs with `probes=[framing]`
- **THEN** the system evaluates model output under different prompt framings and reports the framing-bias score

### Requirement: Probe recency bias
The system SHALL measure whether the model over-weights recently interacted items compared to older interactions.

#### Scenario: Recency bias analysis
- **WHEN** bias analysis runs with `probes=[recency]`
- **THEN** the system computes a recency-bias metric from the temporal distribution of recommended items relative to user history

### Requirement: Configurable probe set
The system SHALL allow the user to select which bias probes to run via the `probes` config list.

#### Scenario: Run subset of probes
- **WHEN** bias analysis runs with `probes=[popularity, position]`
- **THEN** only popularity and position probes execute; framing and recency are skipped
