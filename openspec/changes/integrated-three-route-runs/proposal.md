## Why

Old OpenSpec pipelines (`grpo4rec` / MiniOneRec MovieLens SID / MLLM4Rec) and CLI (`workflow=…`, staged sbatch) no longer match origin/main. Research comparison now runs as three integrated routes via `prepare.sh` + `run.sh`. Remaining old tasks pointed at deleted entrypoints → archived, no delta-spec sync into main.

## What Changes

- Archive (done, no main-spec sync): `minionerec-pipeline`, `mllm4rec-pipeline`.
- Run three self-contained 0.5B experiments via `run.sh` with `WANDB_MODE=online`: MiniOneRec, Rec-R1, DPO4Rec.
- Reuse Amazon Reviews 2023 Industrial raw on scratch (`/scratch/esd4uq/data_2/raw/`). Skip `prepare.sh` download. Re-emit four-file processed contract under `data/processed/amazon23/` (MiniOneRec `data_2/processed/` different layout). Rebuild SID + BM25 under `artifacts/` (don't import `data_2/.../sid/`).
- Launch on dedicated B200 (`sds-rcnode-1` / `dedicated` / `--reservation=sds-rcnode-1-2`), not public `aikyamlab` B200 path.
- Add Slurm wrappers calling `prepare.sh` / `run.sh` instead of deleted `workflow=` CLI.

## Capabilities

### New Capabilities

- `integrated-experiment-runs`: one-prepare-then-three-`run.sh` campaign, stage chains, wandb online, dedicated B200 resource contract.

### Modified Capabilities

- `data-preparation`: `prepare.sh` STEPS, four-file processed contract, skip-if-exists, Amazon23 Industrial raw reuse from `/scratch/esd4uq/data_2/raw/`.

## Impact

- Planning: this change's artifacts replace archived pipeline tasks as run-next source.
- Launchers: new `scripts/*.sbatch` wrap `prepare.sh` / `run.sh`; old `grpo4rec_*` / `minionerec_*` / `mllm4rec_*` sbatch not run path.
- Data: Amazon23 Industrial raw on scratch; SID/BM25 built in new `artifacts/` layout. Training uses `configs/exp/*_qwen05b_amazon.yaml`, no MovieLens override.
- wandb: `run.sh` default `WANDB_MODE=online`; no per-job `wandb login`.
