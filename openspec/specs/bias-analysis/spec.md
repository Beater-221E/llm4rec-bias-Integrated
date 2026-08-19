# bias-analysis Specification

## Purpose

Measure recommendation bias across dimensions (popularity, position, framing, recency) to diagnose fairness issues in trained model.

## Requirements

### Requirement: Probe popularity bias
System SHALL measure how model recommendations correlate with item popularity, report popularity-bias score.

#### Scenario: Popularity bias analysis
- **WHEN** bias analysis runs with `probes=[popularity]`
- **THEN** system computes popularity bias metric comparing recommended items against overall item popularity distribution

### Requirement: Probe position bias
System SHALL measure whether model biased by item position in input sequence.

#### Scenario: Position bias analysis
- **WHEN** bias analysis runs with `probes=[position]`
- **THEN** system reports position-bias score indicating whether earlier or later history items favored

### Requirement: Probe framing bias
System SHALL measure whether recommendations change with prompt framing (e.g., neutral vs. biased phrasing).

#### Scenario: Framing bias analysis
- **WHEN** bias analysis runs with `probes=[framing]`
- **THEN** system evaluates output under different framings, reports framing-bias score

### Requirement: Probe recency bias
System SHALL measure whether model over-weights recently interacted items vs. older interactions.

#### Scenario: Recency bias analysis
- **WHEN** bias analysis runs with `probes=[recency]`
- **THEN** system computes recency-bias metric from temporal distribution of recommended items relative to user history

### Requirement: Configurable probe set
System SHALL let user select bias probes via `probes` config list.

#### Scenario: Run subset of probes
- **WHEN** bias analysis runs with `probes=[popularity, position]`
- **THEN** only popularity + position probes execute; framing + recency skipped
