## 1. Environment Setup

- [x] 1.1 Ensure conda env `bias` active, dependencies installed
- [x] 1.2 Verify `TMDB_API_KEY` set (optional — only needed for full multimodal build)

## 2. Data Preparation (Text-Only Smoke)

- [x] 2.1 Run data build text-only (`python -m llm4rec_bias_Integrated.data.mllm4rec.cli build --config mllm4rec_ml100k --skip-multimodal`)
- [x] 2.2 Verify `data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/dataset.pkl` exists with train/val/test splits + item titles
- [x] 2.3 Verify Parquet serialization output alongside pickle

## 3. Retriever Training

- [x] 3.1 Run smoke retriever training (`python -m llm4rec_bias_Integrated.mllm4rec.cli train-retriever --config mllm4rec_retriever`) with 2 epochs
- [x] 3.2 Verify `experiments/lru/ml-100k/retrieved.pkl` produced with ranked candidates per user
- [x] 3.3 Run full-scale retriever training (500 epochs, early stopping patience=20, validation every 500 iterations)
- [x] 3.4 Verify retriever recall@K meets expectations on validation set

## 4. Ranker Training

- [x] 4.1 Run smoke ranker training (`python -m llm4rec_bias_Integrated.mllm4rec.cli train-ranker --config mllm4rec_ranker --retrieved-pkl experiments/lru/ml-100k/retrieved.pkl`) with 1 epoch / max 20 steps
- [x] 4.2 Verify LoRA adapter saved for Qwen2.5-0.5B ranker
- [x] 4.3 Verify negative sampling (19 negatives per positive) + history truncation (max_history=25) applied
- [ ] 4.4 Run full-scale ranker training (3 epochs, micro_batch_size=2, lr=1e-4)

## 5. Two-Stage Evaluation

- [x] 5.1 Run evaluation on full retriever → ranker pipeline, verify retrieval recall metrics reported
- [x] 5.2 Verify final ranking metrics (NDCG, MRR, Hit Rate) on ranker output
- [x] 5.3 Confirm evaluation identifies cascading errors (gold item missed by retriever → ranker cannot recover)

## 6. Full Multimodal Pipeline (Optional)

- [ ] 6.1 Run full data build with TMDB posters + BLIP2 captions (`python -m llm4rec_bias_Integrated.data.mllm4rec.cli build --config mllm4rec_ml100k`)
- [ ] 6.2 Verify posters downloaded to preprocessing directory with resume support
- [ ] 6.3 Verify BLIP2 captions generated per movie, stored alongside dataset
- [ ] 6.4 Re-run smoke retriever + ranker training with multimodal data, verify prompts include image/caption references

## 7. End-to-End Validation

- [ ] 7.1 Run `./mllm4rec.sh`, verify text-only pipeline completes
- [ ] 7.2 Run `./smoke.sh`, verify mllm4rec portion completes alongside grpo4rec + minionerec
