## Context

Motivation in proposal.md. origin/main has `prepare.sh`, `run.sh`, `configs/exp/*_qwen05b_amazon.yaml`. Rivanna `~/llm4rec-bias-Integrated` on old `main` (`c45cd60`). Amazon23 Industrial raw at `/scratch/esd4uq/data_2/raw/` (645M reviews + 269M meta). Dedicated B200 reservation `sds-rcnode-1-2`: two idle nodes (`udc-ba11-15`, `udc-ba11-23`), 8 B200 each.

## Goals / Non-Goals

**Goals:**
- Sync Rivanna to origin/main `run.sh` entrypoints before launch.
- Prepare once from existing Amazon23 Industrial raw; skip download; rebuild four-file contract + SID/BM25.
- Submit four Slurm jobs: prepare + three trains, `WANDB_MODE=online`, dedicated B200.

**Non-Goals:**
- Re-implement trainers, decoders, bias math.
- Re-download Amazon23 raw (already on scratch).
- Import MiniOneRec `data_2/.../sid/` into new SID hash store.
- MiniOneRec `mode=reproduction` or 7B overlays.
- Public `aikyamlab -p gpu` B200 jobs.

## Decisions

### 1. Dataset: Amazon23 Industrial from scratch raw, not a re-download

Configs default to Amazon23 Industrial. Symlink `/scratch/esd4uq/data_2/raw/{Industrial_and_Scientific,meta_Industrial_and_Scientific}.jsonl.gz` into `data/raw/amazon23/`. MiniOneRec processed CSV/`sid_map.json` under `data_2/processed/` ≠ four-file contract.

**Why:** Raw files match README sizes (644 MB reviews + 268 MB meta). User asked reuse existing data.

**Alternative considered:** MovieLens-100K (`llm4rec-bias-Integrated/data/`). Rejected after scratch `data_2/raw` found.

### 2. GPU N=1 for every job

0.5B full-param SFT ~14 GB, RL ~18 GB; one B200 (~180 GB) enough. Three `N=1` jobs run at once on idle dedicated reservation.

**Why:** Route parallelism beats 4-GPU MiniOneRec when 16 B200s idle.

**Alternative considered:** README `GPUS=0,1,2,3` example for MiniOneRec. Keep as optional override, not default.

### 3. Walltime

| Job | N | Time | Reason |
|---|---|---|---|
| prepare | 1 | `4:00:00` | skip download; data + embed + RQ-VAE SID + BM25 on Industrial |
| minionerec | 1 | `12:00:00` | SFT 3 epochs + RL 2 epochs + two evals |
| recr1 | 1 | `12:00:00` | `max_steps=1000`, `group_size=12`, generate + BM25 per step |
| dpo4rec | 1 | `12:00:00` | reranker + SFT + DPO `T=2`, `N=10` |

**Alternative considered:** 24h everywhere. Keep 12h for 0.5B on B200; Rec-R1 resumes from `save_steps=500` on overrun.

### 4. Launchers wrap `prepare.sh` / `run.sh`

New sbatch files set `PATH` `~/.conda/envs/bias/bin`, `PYTHONPATH=src`, reservation flags, then call `prepare.sh` / `run.sh` with Amazon23 exp configs. Do not revive `python -m llm4rec.cli.main train workflow=…`.

### 5. wandb

Leave `WANDB_MODE=online` (run.sh default). Credentials from `$HOME/.netrc`. No login in sbatch.

## Risks / Trade-offs

**Risk:** Rivanna git behind origin/main; old `workflow=` CLI launch fails.
**Mitigation:** Fetch/checkout origin/main (commit with `run.sh`) on Rivanna before sbatch.

**Risk:** MiniOneRec processed files under `data_2/processed/Industrial_and_Scientific/` ≠ four-file contract.
**Mitigation:** `STEPS=data,embed,sid,bm25`, no `FORCE=1`; data step writes `data/processed/amazon23/Industrial_and_Scientific/` from existing jsonl.gz.

**Risk:** Rec-R1 1000 steps may exceed 12h.
**Mitigation:** `12:00:00` + mid-run checkpoints (`save_steps=500`); resume with `RESUME_FROM` if needed.

## Migration Plan

1. Archive old OpenSpec changes (done; no main-spec sync).
2. Point Rivanna tree at origin/main `run.sh`.
3. Submit prepare, then three trains.
4. Rollback: leave `data_2/` untouched; new outputs to `data/processed/amazon23/`, `artifacts/`, `runs/`.
