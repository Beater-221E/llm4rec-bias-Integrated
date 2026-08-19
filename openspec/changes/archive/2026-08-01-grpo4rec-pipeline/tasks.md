## 1. Environment Setup

- [x] 1.1 Create conda env `bias`, install deps (`pip install -r requirements.txt && pip install -e .`)
- [x] 1.2 Set `PYTHONPATH=src`, validate env (`python -m llm4rec.cli.main validate experiment=smoke_test`)

## 2. Data Preparation

- [x] 2.1 Download classic MovieLens-100K raw (`u.data` / `u.item`) into `data/raw/movielens_100k/`
- [x] 2.2 Run smoke preprocessing (`python -m llm4rec.cli.main prepare experiment=smoke_test dataset=movielens_100k`)
- [x] 2.3 Verify output: train/val/test splits, interaction histories, popularity metadata in `data/processed/movielens_100k/`

## 3. SFT Training

- [x] 3.1 Run smoke SFT (`python -m llm4rec.cli.main train experiment=smoke_grpo stages=[sft]`), verify 4-step completion
- [x] 3.2 Verify LoRA checkpoint save + log output (loss decreasing, checkpoint at `runs/`)
- [x] 3.3 Run full-scale SFT (`python -m llm4rec.cli.main train workflow=grpo4rec scale=full stages=[sft]`), confirm multi-epoch convergence

## 4. GRPO Training

- [x] 4.1 Run smoke GRPO (`python -m llm4rec.cli.main train experiment=smoke_grpo stages=[grpo]`), verify reward computation with exact_match + format_validity
- [x] 4.2 Verify GRPO generates multiple candidates per prompt (`num_generations=2` for smoke), applies invalid_penalty for malformed outputs
- [x] 4.3 Run full-scale GRPO (`python -m llm4rec.cli.main train workflow=grpo4rec scale=full stages=[grpo]`) with rank-aware + popularity-penalty rewards

## 5. Evaluation

- [x] 5.1 Run smoke ranking eval (`python -m llm4rec.cli.main evaluate experiment=smoke_grpo`), verify HR, Recall, NDCG, MRR at top-1/5/10
- [x] 5.2 Run full-scale eval on complete test set
- [x] 5.3 Verify upstream eval integration (`use_upstream_eval=true`) produces expected metrics

## 6. Bias Analysis

- [x] 6.1 Run bias analysis with default probes (`python -m llm4rec.cli.main analyze experiment=smoke_grpo`) — popularity, position, framing, recency
- [x] 6.2 Verify each probe produces numeric score in analysis report
- [x] 6.3 Run bias analysis with probe subset (e.g., `probes=[popularity, recency]`), confirm only selected probes execute

## 7. End-to-End Validation

- [x] 7.1 Run full smoke workflow end-to-end (`python -m llm4rec.cli.main train experiment=smoke_test` = SFT + GRPO + evaluate + analyze + report)
- [x] 7.2 Run `./grpo4rec.sh`, verify completes without errors
- [x] 7.3 Run `make test`, confirm all tests pass
