## 1. Sync Rivanna to origin/main

- [x] 1.1 On Rivanna, update `~/llm4rec-bias-Integrated` to commit with `run.sh` / `prepare.sh` (origin/main)
- [x] 1.2 Confirm `python -m llm4rec.cli.main list` shows `minionerec_qwen05b_amazon`, `recr1_qwen05b_amazon`, `dpo4rec_qwen05b_amazon`

## 2. Launchers

- [x] 2.1 Symlink `/scratch/esd4uq/data_2/raw/Industrial_and_Scientific.jsonl.gz` + `meta_Industrial_and_Scientific.jsonl.gz` into `data/raw/amazon23/`
- [x] 2.2 Add `scripts/prepare_amazon23_dedicated_b200.sbatch`: account `sds-rcnode-1`, partition `dedicated`, `--reservation=sds-rcnode-1-2`, `--gres=gpu:b200:1`, `--time=4:00:00`, `WANDB_MODE=online`, `STEPS=data,embed,sid,bm25` (no download), `EXP=minionerec_qwen05b_amazon`
- [x] 2.3 Add three train sbatch wrapping `run.sh mode=integrated`, `WANDB_MODE=online`, `b200:1` / `12:00:00` for MiniOneRec, Rec-R1, DPO4Rec
- [x] 2.4 Point conda at `$HOME/.conda/envs/bias/bin`, `PYTHONPATH=src`; don't source `/opt/miniconda3`

## 3. Prepare (once)

- [x] 3.1 Submit prepare sbatch after explicit `submit` approval; verify download skipped
- [x] 3.2 Verify four-file contract at `data/processed/amazon23/Industrial_and_Scientific/` (`interactions.jsonl`, `item_meta.json`, `popularity.json`, `stats.json`)
- [x] 3.3 Verify SID under `artifacts/sid/`, BM25 under `artifacts/bm25/`; don't import `/scratch/esd4uq/data_2/processed/Industrial_and_Scientific/sid/`

## 4. Three-route training

- [x] 4.1 After prepare succeeds, submit MiniOneRec `run.sh` (`sft,eval,rl,eval`) on 1×B200 / 12h
- [x] 4.2 Submit Rec-R1 `run.sh` (`sft,eval,rl,eval`) on 1×B200 / 12h
- [x] 4.3 Submit DPO4Rec `run.sh` (`train_reranker,sft,eval,dpo,eval`) on 1×B200 / 12h
- [ ] 4.4 Confirm each job uses one wandb run (`WANDB_MODE=online`), no fresh `wandb login`, writes `runs/.../eval/bias_delta.json`
