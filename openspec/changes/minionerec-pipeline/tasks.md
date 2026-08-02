## 1. Prerequisites

- [x] 1.1 Ensure grpo4rec data pipeline has been run and `data/processed/movielens_100k/` exists with train/val/test splits
- [x] 1.2 Ensure environment is set up (conda env `bias`, `PYTHONPATH=src`, editable install)

## 2. SID Data Preparation

- [x] 2.1 Run SID preparation with smoke scale (`python -m llm4rec.cli.main prepare experiment=smoke_sid`) to build residual K-means codebook (64 codes × 3 levels)
- [x] 2.2 Verify codebook file at `data/processed/movielens_100k/sid/codebook.json` exists and has 64 entries per level
- [x] 2.3 Verify SID mapping files (`sid_train.jsonl`, `sid_val.jsonl`, `sid_test.jsonl`) exist and map all items to 3-token sequences
- [x] 2.4 Verify collision handling: no two items in the mapping share identical SID codes (or are resolved with extra levels)

## 3. SID SFT Training

- [x] 3.1 Run smoke-scale SID SFT training (`python -m llm4rec.cli.main train experiment=smoke_sid stages=[sft]`) with SID-formatted prompts
- [x] 3.2 Verify prompts format items as SID token sequences and targets as SID tokens
- [x] 3.3 Run full-scale SID SFT training and confirm multi-epoch convergence

## 4. SID GRPO Training

- [x] 4.1 Run smoke-scale SID GRPO training (`python -m llm4rec.cli.main train experiment=smoke_sid stages=[grpo]`) with SID-specific parameters (num_generations=2, max_completion_length=8)
- [x] 4.2 Verify GRPO generates SID tokens and decodes them back to items for reward computation
- [ ] 4.3 Verify prefix_credit rewards partial SID matches (correct first token = partial reward)
- [ ] 4.4 Verify invalid_penalty (-0.5) applied to undecodable SID sequences
- [x] 4.5 Run full-scale SID GRPO training with 4 generations and prefix_credit=0.1

## 5. SID Evaluation

- [x] 5.1 Run smoke-scale SID evaluation (`python -m llm4rec.cli.main evaluate experiment=smoke_sid`) and verify ranking metrics
- [x] 5.2 Verify SID validity rate is reported (fraction of generated sequences that decode successfully)
- [ ] 5.3 Verify semantic collision rate is reported
- [ ] 5.4 Verify free-generation evaluation with `free_gen_n=4` (8 for full scale) produces expected generation count
- [ ] 5.5 Run full-scale evaluation with `free_gen_n=50` and top-K ranking metrics

## 6. End-to-End Validation

- [x] 6.1 Run full smoke SID workflow (`python -m llm4rec.cli.main train experiment=smoke_sid`) including SFT + GRPO + evaluation
- [ ] 6.2 Run `./minionerec.sh` convenience script and verify completion
- [ ] 6.3 Run `./smoke.sh` and verify the minionerec portion completes alongside grpo4rec and mllm4rec
