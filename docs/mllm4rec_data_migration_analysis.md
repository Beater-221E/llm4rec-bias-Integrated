# MLLM4Rec Data Migration Analysis (Phase 1)

**Official repo inspected:** [wangyuxiang123/MLLM4Rec](https://github.com/wangyuxiang123/MLLM4Rec) (shallow clone commit at audit time)  
**Target project:** [dragonfly90/llm4rec-bias-Integrated](https://github.com/dragonfly90/llm4rec-bias-Integrated) (`/home/sheng/proj/llm4rec-bias-Integrated`)  
**Scope of this document:** official data-generation call graph, `dataset.pkl` fields, downstream loader dependencies, and migration mapping.  
**Not done in Phase 1:** code changes, full MovieLens downloads, poster downloads, BLIP2 downloads.

**Phase 2 follow-up:** detailed file/symbol → target module mapping is in [`mllm4rec_data_migration_mapping.md`](mllm4rec_data_migration_mapping.md).

**Attribution / license:** Upstream README does not ship a `LICENSE` file; code is “implemented based on [LlamaRec](https://github.com/Yueeeeeeee/LlamaRec)”. Migration must retain README citation and paper cite; do not invent a license. Hardcoded TMDb key in official `process_item.py` must **not** be copied into the target repo.

---

## 4.1 Official data-generation call chain

### A. Preprocess + poster download (`process_item.py`)

```text
process_item.main(args)
  ├─ dataset_factory(args)                    # datasets/__init__.py
  │     └─ DATASETS[args.dataset_code](args)  # e.g. ML100KDataset
  ├─ dataset.load_dataset()                   # datasets/base.py
  │     ├─ preprocess()                       # skip if dataset.pkl exists
  │     │     ├─ maybe_download_raw_dataset()
  │     │     ├─ load_ratings_df()
  │     │     ├─ load_meta_dict()             # (+ meta_img for Amazon)
  │     │     ├─ filter items without meta
  │     │     ├─ filter_triplets()            # iterative min_uc / min_sc
  │     │     ├─ densify_index()              # IDs start at 1; 0 = padding
  │     │     ├─ split_df()                   # leave-one-out
  │     │     └─ pickle.dump → dataset.pkl
  │     └─ pickle.load(dataset.pkl)
  └─ for ml-100k only:
        for key, title in dataset["meta"].items():
            img_path = .../img/{key}.jpg
            if missing:
                find_img(title)               # TMDb search, take movies[0]
                requestPicture(url, path)     # requests.get, no retry
```

Amazon datasets (`beauty` / `toys` / `games`) instead download from `dataset["meta_img_url"]` with `ThreadPoolExecutor(max_workers=128)`.

### B. BLIP2 captions (`process_item_blip2.py`)

```text
process_item_blip2.main(args)
  ├─ dataset_factory(args).load_dataset()     # loads existing dataset.pkl
  ├─ Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=float16)
  ├─ dataset["meta_img_des"] = {}
  ├─ for key in meta:
  │     if img missing → meta_img_des[key] = ""
  │     else:
  │         Blip2Processor.from_pretrained(model_path)   # INSIDE loop (official)
  │         Image.open(img_path)                         # no .convert("RGB")
  │         processor(images=image, return_tensors="pt").to(device, float16)
  │         model.generate(**inputs)                     # NO extra generate kwargs
  │         batch_decode(..., skip_special_tokens=True)[0].strip()
  │         meta_img_des[key] = caption
  └─ pickle.dump(dataset) → same dataset.pkl path (overwrite)
```

### C. Training loaders (downstream consumers)

```text
train_retriever.py
  └─ dataloader_factory(args)  # model_code='lru'
        └─ LRUDataloader(args, dataset_obj)
              └─ dataset.load_dataset()
              uses: train, val, test, umap, smap
              (does NOT read meta / meta_img_des)

train_ranker.py
  └─ dataloader_factory(args)  # model_code='llm'
        └─ LLMDataloader(args, dataset_obj)
              └─ dataset.load_dataset()
              uses: train, val, test, umap, smap, meta, meta_img_des
              + experiments/lru/<dataset_code>/retrieved.pkl
```

---

## 4.2 Official data object structure

### Initial `dataset.pkl` after `ML100KDataset.preprocess()` (before BLIP2)

Built in [`datasets/ml_100k.py`](https://github.com/wangyuxiang123/MLLM4Rec/blob/main/datasets/ml_100k.py):

| Field | Type (as constructed) | Meaning |
|-------|------------------------|---------|
| `train` | `dict[int, list[int]]` | user_id → item sequence excluding last 2 |
| `val` | `dict[int, list[int]]` | user_id → `[item_{-2}]` (length-1 list) |
| `test` | `dict[int, list[int]]` | user_id → `[item_{-1}]` |
| `meta` | `dict[int, str]` | **internal** item_id → cleaned title string **including year suffix** (e.g. `"Toy Story (1995)"` style after their parse) |
| `umap` | `dict[raw_uid → int]` | raw user id → internal user id (starts at **1**) |
| `smap` | `dict[raw_sid → int]` | raw item id → internal item id (starts at **1**) |

**Not present** in ml-100k initial pickle:

- `meta_img_url` (Amazon-only at preprocess time)
- `meta_img_des` (added only by `process_item_blip2.py`)

### After `process_item_blip2.py`

Same dict **plus**:

| Field | Type | Meaning |
|-------|------|---------|
| `meta_img_des` | `dict[int, str]` | internal item_id → BLIP2 caption, or `""` if image missing / blocked |

Official invariant after BLIP2: every `meta` key gets a `meta_img_des` entry (empty string if no image).

### Path layout (hardcoded relative to CWD)

```text
data/                                    # RAW_DATASET_ROOT_FOLDER in config.py
├── ml-100k/                             # raw download folder name = dataset code()
│   └── (ml-latest-small contents!)
└── preprocessed/
    └── ml-100k_min_rating0-min_uc5-min_sc5/
        ├── dataset.pkl
        └── img/
            ├── 1.jpg                    # filename = internal item id
            ├── 2.jpg
            └── ...
```

`min_rating` / `min_uc` / `min_sc` appear in the folder name via `AbstractDataset._get_preprocessed_folder_path()`.

---

## 4.3 Critical official behaviors (must preserve in `compatibility_mode: original`)

### Dataset identity mismatch (HIGH IMPACT)

Despite `code() == "ml-100k"`, official download URL is:

```text
https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
```

Expected files: `movies.csv`, `ratings.csv`, `users.csv` (plus `README`) — **not** classic GroupLens `ml-100k` (`u.data` / `u.item`).

**Implication for migration:**

- `original` mode must reproduce **ml-latest-small** under the `ml-100k` dataset code (official naming).
- Classic GroupLens ML-100K (`u.data`) and ML-1M are **extensions** / separate codes unless explicitly aliased; do not claim they are byte-identical to official `ml-100k` runs.

### Filtering

[`filter_triplets`](https://github.com/wangyuxiang123/MLLM4Rec/blob/main/datasets/base.py):

- Defaults: `min_uc=5`, `min_sc=5`, `min_rating=0`.
- **`min_rating` is stored and used in path names but never applied as a rating filter** in `filter_triplets`.
- Filtering is **iterative**: repeatedly drop items with count `< min_sc` and users with count `< min_uc` until stable (k-core style loop).
- Assert: `min_uc >= 2`.

### ID mapping

- `densify_index`: `enumerate(set(...), start=1)` → internal IDs in `{1..N}`.
- Padding token **0** is reserved (see `LRUTrainDataset` left-pad with `0`).
- `meta` keys are **internal** item IDs after remap (`meta = {smap[k]: v for k,v in meta_raw.items() if k in smap}`).
- Image filenames use the same internal keys: `{key}.jpg`.

### Sorting & split

- Per-user sequence: `sort_values(by=['timestamp', 'sid'])` then take `sid` list.  
  (Second key is `sid`, **not** original row index — official behavior.)
- Leave-one-out:
  - `train = items[:-2]`
  - `val = items[-2:-1]`
  - `test = items[-1:]`
- Users keyed as `1 .. user_count` after densify.

### Movie title → TMDb (`find_img`)

```python
movies = tmdb.search().movies(movie_name[:-7])  # strip last 7 chars (year)
movie_id = movies[0].id                         # ALWAYS first hit
img_url = "https://image.tmdb.org/t/p/original/" + movie.poster_path
```

- Hardcoded TMDb key in source (must be replaced by env var in target).
- Bare `except:` → return `""`.
- Missing poster / failed search: **skip download**, do **not** delete item from pickle.
- Does **not** write URLs back into `dataset.pkl`.

### BLIP2 generate (official kwargs)

Exactly:

```python
inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs)
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
```

No prompt, no `max_new_tokens`, no beams, no sampling kwargs.  
Processor is reloaded **inside** the per-item loop.  
`Image.open` without `.convert("RGB")`.  
Default model path is CLI `--model_path` (README: `Salesforce/blip2-opt-2.7b`).  
`games` dataset has a special SHA256 skip for a known bad image (`utils.encrypt`).

### Title cleaning for `meta` (ml-100k / ml-latest-small)

From `load_meta_dict`:

1. Read `movies.csv` with `encoding="ISO-8859-1"`.
2. `title = row[2][:-7]`; `year = row[2][-7:]`.
3. `re.sub('\\(.*?\\)', '', title).strip()`.
4. Optional article move if title ends with `, a|an|the`.
5. Store `meta_raw[movieId] = title + year` (year suffix reattached).

---

## 4.3 Downstream field dependencies

### `train_retriever.py` / `LRUDataloader`

Reads from pickle:

- `train`, `val`, `test`, `umap`, `smap`

Sets `args.num_users`, `args.num_items` from map lengths.  
Padding item id `0`. Does **not** need images or captions.

### `train_ranker.py` / `LLMDataloader`

Reads from pickle:

- `train`, `val`, `test`, `umap`, `smap`
- `meta` → `text_dict` (titles)
- `meta_img_des` → `text_img_dict` (captions) — **required**; KeyError if missing

Prompt construction (`seq_to_token_ids`) formats each item as:

```text
{truncated_title : truncated_caption}
```

Also requires `retrieved.pkl` from retriever stage (not part of data generation).

---

## 4.4 Directly portable vs needs adaptation

| Category | Items |
|----------|--------|
| **Can migrate nearly as-is** | `filter_triplets`, `densify_index`, `split_df`, `load_meta_dict` title rules, leave-one-out semantics, BLIP2 generate call shape, img naming `{id}.jpg`, folder naming pattern |
| **Import / packaging changes** | `from datasets import …` → package imports; remove `sys.path` / CWD-relative `./data/...` |
| **Path / config abstraction** | `RAW_DATASET_ROOT_FOLDER='data'`; hardcoded `./data/preprocessed/...`; device strings; `model_path` |
| **Must rewrite as modules** | `process_item.py` + `process_item_blip2.py` scripts → staged CLI; TMDb client; poster downloader; captioner; serializer; validator |
| **Security / secrets** | Hardcoded TMDb API key in `find_img` — **do not copy**; use `TMDB_API_KEY` env |
| **Official defects to preserve in `original`** | (1) `min_rating` unused as filter; (2) TMDb always `movies[0]`; (3) bare excepts; (4) no download retry; (5) BLIP2 processor loaded per item; (6) no RGB convert; (7) dataset code `ml-100k` ≠ classic ml-100k zip; (8) `process_item` does not update pickle with poster URLs; (9) sort secondary key `sid` not row index |
| **Robust-only enhancements** | retries, caches (`tmdb_matches.jsonl`, `captions.jsonl`), year-aware TMDb match, PIL verify, atomic pickle write, batch caption, parquet sidecars, validation reports |

---

## Migration mapping (for Phase 2+)

| Official file / symbol | Target (planned under `llm4rec-bias-Integrated`) | Mode |
|------------------------|----------------------------------------|------|
| `datasets/__init__.py::dataset_factory` | `…/mllm4rec/dataset_factory.py` | adapt imports |
| `datasets/base.py::AbstractDataset` | `…/mllm4rec/base_dataset.py` | migrate + config paths |
| `datasets/ml_100k.py::ML100KDataset` | `…/mllm4rec/movielens_100k.py` | **original = ml-latest-small under code `ml-100k`** |
| (new) classic GroupLens + ML-1M | `movielens_100k_classic.py` / `movielens_1m.py` | extension / robust docs |
| `datasets/base.py::filter_triplets` | `filtering.py` | original iterative k-core |
| `densify_index` | `id_mapping.py` | start=1, keep 0 pad |
| `split_df` | `splitting.py` | LOO as official |
| `process_item.find_img` / `requestPicture` | `tmdb_client.py`, `poster_downloader.py` | original vs robust |
| `process_item_blip2.main` loop | `blip2_captioner.py` | original generate kwargs |
| pickle dump/load | `serializer.py` | official schema + atomic/backup in robust |
| schema checks | `validator.py`, `compatibility.py` | both modes |
| CLI scripts | `cli.py` + `scripts/mllm4rec/*` | staged pipeline |

**Package path note:** Task sketch uses `src/llm4rec/data/mllm4rec/`. This repo’s installable package is `llm4rec_bias_Integrated`. Phase 2 should place modules under `src/llm4rec_bias_Integrated/data/mllm4rec/` (or an agreed alias) so `python -m llm4rec_bias_Integrated.data.mllm4rec.cli` works without breaking existing letter/SID workflows.

---

## Integration with existing `llm4rec-bias-Integrated` data stack

Existing lab already has:

- MovieLens-100K / 1M adapters under `datasets/movielens/` (classic GroupLens files, LOO, string item ids, rating threshold, etc.)
- SID path under `semantic_ids/`

**Rules for later phases:**

1. Do **not** change letter-route / SID schemas for MLLM4Rec.
2. Keep MLLM4Rec official pickle under `data/preprocessed/.../dataset.pkl` as a **separate** workflow artifact.
3. If classic GroupLens ML-100K is needed for paper-style “MovieLens-100K” experiments in this lab, expose it as an explicit non-`original` dataset code (e.g. `ml-100k-classic`) so it is not confused with official `ml-100k` (= ml-latest-small).

---

## Phase 1 report

```text
Phase: 1 — Official code inspection
Official code inspected: YES (README, config.py, process_item.py, process_item_blip2.py,
  datasets/{__init__,base,ml_100k,utils,beauty}, dataloader/{__init__,lru,llm},
  train_retriever.py, train_ranker.py)
Files created: docs/mllm4rec_data_migration_analysis.md
Files modified: (none)
Functions migrated: (none)
Behavior preserved: N/A (analysis only)
Engineering changes: none
Commands executed: git clone --depth 1 https://github.com/wangyuxiang123/MLLM4Rec.git (temp)
Tests passed: N/A
Tests failed: N/A
Known issues / official quirks documented:
  - dataset_code ml-100k downloads ml-latest-small, not classic ml-100k
  - min_rating unused as filter
  - TMDb key hardcoded; first search hit only
  - meta_img_des only after BLIP2; LRU does not need it, LLM ranker does
  - BLIP2 processor constructed inside item loop; generate() with default kwargs only
Next action: Phase 2 — design migration mapping table in implementation plan /
  then Phase 3 MovieLens preprocess without TMDb/BLIP2
```

**Explicitly not started (per Phase 1 constraints):** full dataset download, poster crawl, BLIP2 model download, target-repo code edits beyond this document.
