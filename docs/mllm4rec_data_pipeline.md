# MLLM4Rec Data Pipeline Guide

Official-compatible multimodal data generation for [MLLM4Rec](https://github.com/wangyuxiang123/MLLM4Rec), integrated in `llm4rec-bias-Integrated`.

Design notes: [mllm4rec_data_migration_analysis.md](mllm4rec_data_migration_analysis.md), [mllm4rec_data_migration_mapping.md](mllm4rec_data_migration_mapping.md), [mllm4rec_data_compatibility_report.md](mllm4rec_data_compatibility_report.md).

## 1. Flow

```text
download → preprocess → match-tmdb → download-posters → generate-captions → serialize → validate
```

## 2. Official sources

Adapted from `datasets/ml_100k.py`, `datasets/base.py`, `process_item.py`, `process_item_blip2.py`.

## 3. Module layout

```text
src/llm4rec_bias_Integrated/data/mllm4rec/
configs/dataset/mllm4rec_ml100k.yaml
configs/dataset/mllm4rec_ml1m.yaml
scripts/mllm4rec/
tests/data/mllm4rec/
```

## 4. Environment

```bash
conda activate bias          # torch 2.13+cu126 for V100
cd llm4rec-bias-Integrated
pip install -r requirements.txt
pip install -e .
export TMDB_API_KEY="..."    # required for TMDb / posters
```

## 5. MovieLens-100K (official-compatible = ml-latest-small)

```bash
# Text-only
PYTHONPATH=src python -m llm4rec_bias_Integrated.data.mllm4rec.cli build \
  --config configs/dataset/mllm4rec_ml100k.yaml --skip-multimodal

# Full multimodal (or staged commands below)
export TMDB_API_KEY="..."
PYTHONPATH=src python -m llm4rec_bias_Integrated.data.mllm4rec.cli match-tmdb \
  --config configs/dataset/mllm4rec_ml100k.yaml
PYTHONPATH=src python -m llm4rec_bias_Integrated.data.mllm4rec.cli download-posters \
  --config configs/dataset/mllm4rec_ml100k.yaml
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m llm4rec_bias_Integrated.data.mllm4rec.cli generate-captions \
  --config configs/dataset/mllm4rec_ml100k.yaml
PYTHONPATH=src python -m llm4rec_bias_Integrated.data.mllm4rec.cli validate \
  --config configs/dataset/mllm4rec_ml100k.yaml --require-captions
```

## 6. MovieLens-1M (extension)

```bash
PYTHONPATH=src python -m llm4rec_bias_Integrated.data.mllm4rec.cli build \
  --config configs/dataset/mllm4rec_ml1m.yaml --skip-multimodal
# Then same multimodal stages with mllm4rec_ml1m.yaml
```

## 7. TMDb key

```bash
export TMDB_API_KEY="..."
# or put export in ~/.bashrc (before the interactive-only early return)
```

Missing key → clear CLI error. Never commit keys.

## 8–9. Posters & BLIP2

- Posters: `img/{internal_item_id}.jpg`; failures → `failed_posters.jsonl`; items kept.
- Default model: `Salesforce/blip2-opt-2.7b`, `caption.dtype: float16` on V100.
- Captions resume via `captions.jsonl`; missing image → `""`.

## 10–11. Commands

See root [README.md](../README.md) §5. Staged CLI subcommands: `download`, `preprocess`, `match-tmdb`, `download-posters`, `generate-captions`, `serialize`, `validate`, `build`.

## 12. Resume

Default `--resume`. Re-run the same stage; completed TMDb rows / existing jpgs / caption lines are skipped. Use `--overwrite` to force redo.

## 13. Outputs

```text
data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/
├── dataset.pkl
├── img/
├── tmdb_matches.jsonl
├── captions.jsonl
├── user_id_map.json
├── item_id_map.json
├── validation_report.json
└── *.parquet
```

## 14. Validation

```bash
PYTHONPATH=src python -m llm4rec_bias_Integrated.data.mllm4rec.cli validate \
  --config configs/dataset/mllm4rec_ml100k.yaml --require-captions
python -m pytest tests/data/mllm4rec -q
```

## 15. original vs robust

| | original (default) | robust |
|--|-------------------|--------|
| TMDb | first search hit | year ±1 preference |
| `min_rating` | path only | optional rating filter |
| sort tie-break | `timestamp,sid` | optional `source_row_index` |

## 16. FAQ

- **`nvidia-smi` NVML mismatch:** use `conda activate bias`; PyTorch may still work. Reboot to fix NVML.
- **Wrong torch (cu130) on V100:** always use `bias` env (`cu126`).
- **Toy Story → Toy Story 5:** original first-hit TMDb; use `match_mode: robust` to improve.
- **Empty captions:** missing posters; expected; do not drop items.

## 17. V100 FP16

```yaml
caption:
  device: cuda
  dtype: float16
```

Do not default to bf16 on V100.

## 18. Reproduction vs engineering

Official reproduction = pickle schema + original filter/split/TMDb/generate semantics.  
Sidecars, tqdm, retries, processor-once, parquet are engineering only.
