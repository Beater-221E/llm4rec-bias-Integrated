## Purpose

Defines how three integrated LLM4Rec routes prepare + train as one comparable campaign via `prepare.sh` / `run.sh`, incl. wandb online logging + dedicated B200 launch resources.

## ADDED Requirements

### Requirement: Three-route integrated campaign
System SHALL run MiniOneRec, Rec-R1, DPO4Rec as three independent `run.sh` jobs sharing one prepared dataset + same 0.5B Instruct backbone.

#### Scenario: MiniOneRec stages
- **WHEN** `EXP=minionerec_qwen05b_amazon` launched with `mode=integrated`
- **THEN** job executes `sft → eval → rl → eval` without manual stage handoff

#### Scenario: Rec-R1 stages
- **WHEN** `EXP=recr1_qwen05b_amazon` launched with `mode=integrated`
- **THEN** job executes `sft → eval → rl → eval` without manual stage handoff

#### Scenario: DPO4Rec stages
- **WHEN** `EXP=dpo4rec_qwen05b_amazon` launched with `mode=integrated`
- **THEN** job executes `train_reranker → sft → eval → dpo → eval` without manual stage handoff

### Requirement: Single wandb run per experiment
Each `run.sh` job SHALL keep `WANDB_MODE=online`, log all stages into one wandb run. Prior `wandb login` on same Rivanna user account SHALL be reused; job MUST NOT require interactive login.

#### Scenario: Online logging with existing credentials
- **WHEN** `~/.netrc` already has wandb credentials and `WANDB_MODE` is `online`
- **THEN** training initializes wandb run, still writes `runs/.../metrics.jsonl`

#### Scenario: wandb failure does not stop training
- **WHEN** wandb init or logging fails
- **THEN** training continues, metrics remain in `metrics.jsonl`

### Requirement: Dedicated B200 launch contract
Campaign jobs SHALL request GPUs on dedicated B200 reservation (`-A sds-rcnode-1 -p dedicated --reservation=sds-rcnode-1-2 --gres=gpu:b200:<N>`), not public `aikyamlab` gpu partition.

#### Scenario: Prepare job resources
- **WHEN** prepare job submitted
- **THEN** requests `N=1` B200, walltime `4:00:00`

#### Scenario: Training job resources
- **WHEN** MiniOneRec, Rec-R1, or DPO4Rec training job submitted
- **THEN** requests `N=1` B200, walltime `12:00:00`

### Requirement: Shared prepare then independent trains
Campaign SHALL run `prepare.sh` once, then submit three training jobs independently to overlap on unused dedicated B200 GPUs.

#### Scenario: Prepare before train
- **WHEN** training job starts
- **THEN** four-file processed contract, SID artifacts (MiniOneRec), BM25 index (Rec-R1) for that route already exist from prepare, or job fails instead of silently rebuilding SID
