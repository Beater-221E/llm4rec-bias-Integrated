# MLLM4Rec Data Compatibility Report (MovieLens / ml-100k)

**Generated against:** `data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/`  
**Official reference:** [wangyuxiang123/MLLM4Rec](https://github.com/wangyuxiang123/MLLM4Rec)  
**Target:** `llm4rec-bias-Integrated` (`llm4rec_bias_Integrated.data.mllm4rec`)

## Summary table

| 项目 | 官方实现 | 移植实现 | 状态 |
|------|----------|----------|------|
| 数据目录 | `data/preprocessed/ml-100k_min_rating{R}-min_uc{U}-min_sc{S}/` | 相同模板（可配置 `output_root`） | PASS |
| `dataset.pkl` 字段 | `train,val,test,meta,umap,smap` + 后处理 `meta_img_des` | 相同 | PASS |
| `meta` 类型 | `dict[int,str]`（内部 item id → 标题） | 相同 | PASS |
| `meta_img_des` 类型 | `dict[int,str]` | 相同；与 `meta` key 集合一致 | PASS |
| ID 起点 | densify `start=1`；`0` padding | 相同 | PASS |
| 图片文件命名 | `img/{internal_id}.jpg` | 相同 | PASS |
| BLIP2 模型 | `Salesforce/blip2-opt-2.7b` | 相同（YAML） | PASS |
| `generate` 参数 | `model.generate(**inputs)` 无额外采样参数 | 相同 | PASS |
| 缺图处理 | caption `""`，不删 item | 相同（空 caption 24 / 3650） | PASS |
| split 逻辑 | LOO：`[:-2]` / `[-2:-1]` / `[-1:]`；排序 `timestamp,sid` | 相同 | PASS |
| Ranker loader 可读 | 需 `meta` + `meta_img_des` | `simulate_official_ranker_prompt` PASS | PASS |
| Retriever loader 可读 | 需 `train/val/test/umap/smap` | `simulate_official_retriever_load` PASS | PASS |

## Measured coverage (this host)

| Metric | Value |
|--------|------:|
| users | 610 |
| items / meta | 3650 |
| TMDb cache rows | 3650 |
| TMDb matched (status) | 3637 |
| posters downloaded (valid) | 3626 |
| nonempty captions | 3626 |
| empty captions (missing poster) | 24 |
| `meta` ≡ `meta_img_des` keys | yes |
| schema validate `--require-captions` | ok |

## Known semantic differences (documented, not silent fixes)

### 1. Dataset identity: `ml-100k` = ml-latest-small

Official `ML100KDataset` downloads **ml-latest-small**, not classic GroupLens `u.data`.  
Our `dataset.code: ml-100k` preserves that. Classic files live under `ml-100k-classic` (extension).

### 2. TMDb first-hit matching (original)

Official `find_img` takes the **first** search result after `title[:-7]`.  
Example observed: `Toy Story (1995)` → TMDb `Toy Story 5`.  
This is **original-compatible**, not a bug. Switch to year-aware matching with:

```yaml
tmdb:
  match_mode: robust
```

### 3. `min_rating` unused as a filter

Official stores `min_rating` in the folder name only. Original mode does the same.

### 4. BLIP2 processor load

Official reloads `Blip2Processor` inside the per-item loop.  
We load the processor **once** for throughput; `generate(**inputs)` kwargs remain official-equivalent.  
If strict loop parity is required for an ablation, document it as an engineering delta (does not change caption decoding API).

### 5. Engineering sidecars

We also write `tmdb_matches.jsonl`, `captions.jsonl`, parquet maps, validation reports.  
These do **not** replace or alter official pickle fields.

## How to switch back to strict original behavior

```yaml
compatibility:
  mode: original
filtering:
  mode: original
tmdb:
  match_mode: original
caption:
  mode: original
  dtype: float16   # V100
```

Do not enable robust TMDb / rating filters if claiming official reproduction.

## Downstream not in this report

Official Ranker also needs `experiments/lru/<dataset_code>/retrieved.pkl` from Retriever training.  
That artifact is **out of scope** for the data-generation migration.
