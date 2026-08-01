## 1. Environment Setup

- [x] 1.1 Ensure conda env `bias` is active and dependencies are installed
- [x] 1.2 Verify `TMDB_API_KEY` environment variable is set (optional — required only for full multimodal build)

## 2. Data Preparation (Text-Only Smoke)

- [x] 2.1 Run data build in text-only mode (`python -m llm4rec_bias_Integrated.data.mllm4rec.cli build --config mllm4rec_ml100k --skip-multimodal`)
- [x] 2.2 Verify `data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/dataset.pkl` exists with train/val/test splits and item titles
- [x] 2.3 Verify Parquet serialization output exists alongside the pickle file

## 3. Retriever Training

- [x] 3.1 Run smoke-scale retriever training (`python -m llm4rec_bias_Integrated.mllm4rec.cli train-retriever --config mllm4rec_retriever`) with 2 epochs
- [x] 3.2 Verify `experiments/lru/ml-100k/retrieved.pkl` is produced with ranked candidates per user
- [x] 3.3 Run full-scale retriever training (500 epochs, early stopping patience=20, validation every 500 iterations)
- [x] 3.4 Verify retriever recall@K meets expectations on validation set

## 4. Ranker Training

- [x] 4.1 Run smoke-scale ranker training (`python -m llm4rec_bias_Integrated.mllm4rec.cli train-ranker --config mllm4rec_ranker --retrieved-pkl experiments/lru/ml-100k/retrieved.pkl`) with 1 epoch / max 20 steps
- [x] 4.2 Verify LoRA adapter is saved for the Qwen2.5-0.5B ranker
- [x] 4.3 Verify negative sampling (19 negatives per positive) and history truncation (max_history=25) are applied
- [ ] 4.4 Run full-scale ranker training (3 epochs, micro_batch_size=2, lr=1e-4)

## 5. Two-Stage Evaluation

- [x] 5.1 Run evaluation on the full retriever → ranker pipeline and verify retrieval recall metrics are reported
- [x] 5.2 Verify final ranking metrics (NDCG, MRR, Hit Rate) on ranker output
- [x] 5.3 Confirm evaluation identifies cascading errors (gold item missed by retriever → ranker cannot recover)

## 6. Full Multimodal Pipeline (Optional)

- [ ] 6.1 Run full data build with TMDB posters and BLIP2 captions (`python -m llm4rec_bias_Integrated.data.mllm4rec.cli build --config mllm4rec_ml100k`)
- [ ] 6.2 Verify poster images are downloaded to the preprocessing directory with resume support
- [ ] 6.3 Verify BLIP2 captions are generated per movie and stored alongside the dataset
- [ ] 6.4 Re-run smoke retriever + ranker training with multimodal data and verify prompts include image/caption references

## 7. End-to-End Validation

- [ ] 7.1 Run `./mllm4rec.sh` convenience script and verify text-only pipeline completes
- [ ] 7.2 Run `./smoke.sh` and verify the mllm4rec portion completes alongside grpo4rec and minionerec
