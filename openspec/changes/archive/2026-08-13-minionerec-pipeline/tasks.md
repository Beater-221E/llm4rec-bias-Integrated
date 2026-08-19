## 1. Prerequisites

- [x] 1.1 Ensure grpo4rec data pipeline run, `data/processed/movielens_100k/` exists with train/val/test splits
- [x] 1.2 Ensure env set up (conda env `bias`, `PYTHONPATH=src`, editable install)

## 2. SID Data Preparation

- [x] 2.1 Run SID prep with smoke scale (`python -m llm4rec.cli.main prepare experiment=smoke_sid`) — build residual K-means codebook (64 codes × 3 levels)
- [x] 2.2 Verify codebook at `data/processed/movielens_100k/sid/codebook.json`, 64 entries per level
- [x] 2.3 Verify SID mapping files (`sid_train.jsonl`, `sid_val.jsonl`, `sid_test.jsonl`) map all items to 3-token sequences
- [x] 2.4 Verify collision handling: no two items share identical SID codes (or resolved with extra levels)

## 3. SID SFT Training

- [x] 3.1 Run smoke SID SFT (`python -m llm4rec.cli.main train experiment=smoke_sid stages=[sft]`) with SID-formatted prompts
- [x] 3.2 Verify prompts format items as SID token sequences, targets as SID tokens
- [x] 3.3 Run full-scale SID SFT, confirm multi-epoch convergence

## 4. SID GRPO Training

- [x] 4.1 Run smoke SID GRPO (`python -m llm4rec.cli.main train experiment=smoke_sid stages=[grpo]`) with SID params (num_generations=2, max_completion_length=8)
- [x] 4.2 Verify GRPO generates SID tokens, decodes to items for reward computation
- [ ] 4.3 Verify prefix_credit rewards partial SID matches (correct first token = partial reward)
- [ ] 4.4 Verify invalid_penalty (-0.5) applied to undecodable SID sequences
- [x] 4.5 Run full-scale SID GRPO with 4 generations, prefix_credit=0.1

## 5. SID Evaluation

- [x] 5.1 Run smoke SID eval (`python -m llm4rec.cli.main evaluate experiment=smoke_sid`), verify ranking metrics
- [x] 5.2 Verify SID validity rate reported (fraction of generated sequences decoding successfully)
- [ ] 5.3 Verify semantic collision rate reported
- [ ] 5.4 Verify free-generation eval with `free_gen_n=4` (8 for full scale) produces expected generation count
- [ ] 5.5 Run full-scale eval with `free_gen_n=50` + top-K ranking metrics

## 6. End-to-End Validation

- [x] 6.1 Run full smoke SID workflow (`python -m llm4rec.cli.main train experiment=smoke_sid`) — SFT + GRPO + evaluation
- [ ] 6.2 Run `./minionerec.sh`, verify completion
- [ ] 6.3 Run `./smoke.sh`, verify minionerec portion completes alongside grpo4rec + mllm4rec
