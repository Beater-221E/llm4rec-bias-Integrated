## 1. Environment Setup

- [x] 1.1 Create conda environment `bias` and install dependencies (`pip install -r requirements.txt && pip install -e .`)
- [x] 1.2 Set `PYTHONPATH=src` and validate environment (`python -m llm4rec.cli.main validate experiment=smoke_test`)

## 2. Data Preparation

- [x] 2.1 Download classic MovieLens-100K raw data (`u.data` / `u.item`) into `data/raw/movielens_100k/`
- [x] 2.2 Run preprocessing with smoke scale (`python -m llm4rec.cli.main prepare experiment=smoke_test dataset=movielens_100k`)
- [x] 2.3 Verify processed output: train/val/test splits, interaction histories, popularity metadata in `data/processed/movielens_100k/`

## 3. SFT Training

- [x] 3.1 Run smoke-scale SFT training (`python -m llm4rec.cli.main train experiment=smoke_grpo stages=[sft]`) and verify 4-step completion
- [x] 3.2 Verify LoRA checkpoint save and log output (loss decreasing, checkpoint at `runs/`)
- [x] 3.3 Run full-scale SFT training (`python -m llm4rec.cli.main train workflow=grpo4rec scale=full stages=[sft]`) and confirm multi-epoch convergence

## 4. GRPO Training

- [x] 4.1 Run smoke-scale GRPO training (`python -m llm4rec.cli.main train experiment=smoke_grpo stages=[grpo]`) and verify reward computation with exact_match + format_validity
- [x] 4.2 Verify GRPO generates multiple candidates per prompt (`num_generations=2` for smoke) and applies invalid_penalty for malformed outputs
- [x] 4.3 Run full-scale GRPO training (`python -m llm4rec.cli.main train workflow=grpo4rec scale=full stages=[grpo]`) with rank-aware and popularity-penalty rewards

## 5. Evaluation

- [x] 5.1 Run smoke-scale ranking evaluation (`python -m llm4rec.cli.main evaluate experiment=smoke_grpo`) and verify HR, Recall, NDCG, MRR at top-1/5/10
- [x] 5.2 Run full-scale evaluation on complete test set
- [x] 5.3 Verify upstream evaluation integration (`use_upstream_eval=true`) produces metrics matching expectation

## 6. Bias Analysis

- [x] 6.1 Run bias analysis with default probes (`python -m llm4rec.cli.main analyze experiment=smoke_grpo`) — popularity, position, framing, recency
- [x] 6.2 Verify each probe produces a numeric score in the analysis report
- [x] 6.3 Run bias analysis with a subset of probes (e.g., `probes=[popularity, recency]`) and confirm only selected probes execute

## 7. End-to-End Validation

- [x] 7.1 Run full smoke workflow end-to-end (`python -m llm4rec.cli.main train experiment=smoke_test` includes SFT + GRPO + evaluate + analyze + report)
- [x] 7.2 Run `./grpo4rec.sh` convenience script and verify it completes without errors
- [x] 7.3 Run `make test` and confirm all tests pass
