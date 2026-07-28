# MLLM4Rec Data Migration Mapping (Phase 2)

**Depends on:** [`mllm4rec_data_migration_analysis.md`](mllm4rec_data_migration_analysis.md) (Phase 1)  
**Target package:** `llm4rec_bias_Integrated` under `/home/sheng/proj/llm4rec-bias-Integrated`  
**Phase 2 scope:** lock file/function → target module mapping, reuse vs isolate decisions, original/robust boundaries. **No implementation in this phase.**

---

## 1. Package layout decision

Task sketch used `src/llm4rec/data/mllm4rec/`. This repo’s installable package is **`llm4rec_bias_Integrated`**.

| Role | Path |
|------|------|
| Package root | `src/llm4rec_bias_Integrated/data/mllm4rec/` |
| CLI entry | `python -m llm4rec_bias_Integrated.data.mllm4rec.cli` |
| Configs | `configs/dataset/mllm4rec_ml100k.yaml`, `mllm4rec_ml1m.yaml` |
| Shell wrappers | `scripts/mllm4rec/*.sh` |
| Tests | `tests/data/mllm4rec/*.py` |
| Docs | `docs/mllm4rec_*.md` |
| Artifacts (official-compatible) | `{data_root}/preprocessed/{code}_min_rating{R}-min_uc{U}-min_sc{S}/` |
| Existing letter/SID data (unchanged) | `{data_root}/raw|processed|cache/movielens_*` via `datasets/movielens/` |

**Do not** fold MLLM4Rec pickle semantics into `DatasetAdapter` / letter-route schemas.

---

## 2. Dataset identity mapping

| Code in target | Raw source | Compatibility claim | Notes |
|----------------|------------|---------------------|-------|
| `ml-100k` | GroupLens **ml-latest-small** (`ratings.csv`, `movies.csv`) | **official-compatible** (`compatibility.mode: original`) | Matches MLLM4Rec `ML100KDataset.code()` |
| `ml-100k-classic` | GroupLens **ml-100k** (`u.data`, `u.item`, …) | **extension** | Same pipeline abstractions; not byte-identical to official |
| `ml-1m` | GroupLens **ml-1m** (`ratings.dat`, `movies.dat`, …) | **MLLM4Rec-compatible extension** | Reuse base classes; discontinuous MovieIDs via densify |

YAML `dataset.code` drives folder names exactly as official:  
`{code}_min_rating{min_rating}-min_uc{min_uc}-min_sc{min_sc}`.

Default CLI/config for reproduction: **`ml-100k` + original**.

---

## 3. Full symbol → target mapping

### 3.1 Core dataset / preprocess

| Official file | Official symbol | Target file | Migrate how | Keep as-is? | Adaptations |
|---------------|-----------------|-------------|--------------|-------------|-------------|
| `datasets/__init__.py` | `DATASETS`, `dataset_factory` | `dataset_factory.py` | rewrite registry | behavior | package imports; register `ml-100k`, `ml-100k-classic`, `ml-1m` |
| `datasets/base.py` | `AbstractDataset` | `base_dataset.py` | migrate class | mostly | inject config paths; drop `from config import …` |
| `datasets/base.py` | `load_dataset` | `base_dataset.py` / `serializer.py` | migrate | yes | optional atomic load in robust |
| `datasets/base.py` | `filter_triplets` | `filtering.py` | migrate verbatim for original | **yes for original** | robust may expose same iterative loop as named mode; no silent one-pass k-core swap |
| `datasets/base.py` | `densify_index` | `id_mapping.py` | migrate | **yes** (`start=1`) | export JSON maps as sidecars |
| `datasets/base.py` | `split_df` | `splitting.py` | migrate | **yes** | sort keys `timestamp`,`sid` in original; optional `source_row_index` only in robust |
| `datasets/base.py` | `_get_*_path` | `constants.py` + `base_dataset.py` | abstract | path formula | config `output_root`, no CWD hardcode |
| `datasets/ml_100k.py` | `ML100KDataset` | `movielens_100k.py` | migrate | download/parse/meta | code stays `ml-100k`; URL = ml-latest-small |
| `datasets/ml_100k.py` | `load_ratings_df` | `movielens_100k.py` | migrate | yes | columns `uid,sid,rating,timestamp` |
| `datasets/ml_100k.py` | `load_meta_dict` | `movielens_100k.py` | migrate title rules | **yes for original** | robust title query separate helper |
| `datasets/utils.py` | `download`, `unzip` | reuse / thin wrap in `base_dataset.py` | migrate needed bits | behavior | use project download helpers if any; else port |
| — | — | `movielens_1m.py` | **new** | n/a | `::` parser; same filter/densify/split |
| — | — | classic parser in `movielens_100k.py` or sibling | **new** | n/a | ISO-8859-1 `u.item`; extension only |

### 3.2 TMDb + posters

| Official file | Official symbol | Target file | Migrate how | Keep as-is? | Adaptations |
|---------------|-----------------|-------------|--------------|-------------|-------------|
| `process_item.py` | `find_img` | `tmdb_client.py` | extract | original: `name[:-7]`, `movies[0]` | key from `os.environ[TMDB_API_KEY]`; never commit key; robust year match |
| `process_item.py` | `requestPicture` | `poster_downloader.py` | extract | original: no retry, skip if exists | robust: timeout/retry/PIL/atomic rename |
| `process_item.py` | `main` ml-100k branch | `cli.py` `match-tmdb` / `download-posters` | stage split | semantics | cache `tmdb_matches.jsonl`; `--max-items`; failure log |
| `process_item.py` | Amazon thread pool | **out of scope** (optional later) | skip for Phase 3–8 MovieLens focus | — | document only |

### 3.3 BLIP2 captions

| Official file | Official symbol | Target file | Migrate how | Keep as-is? | Adaptations |
|---------------|-----------------|-------------|--------------|-------------|-------------|
| `process_item_blip2.py` | model load `float16` | `blip2_captioner.py` | migrate | **yes** | `model_name_or_path` from config; device from config |
| `process_item_blip2.py` | per-item processor + `generate(**inputs)` | `blip2_captioner.py` | migrate | **original = defaults only**, processor-in-loop | robust: load processor once; optional batch; still empty string on miss |
| `process_item_blip2.py` | missing img → `""` | `blip2_captioner.py` | migrate | **yes** | never drop items |
| `process_item_blip2.py` | rewrite full pickle | `serializer.py` + captioner | migrate | schema | robust: `captions.jsonl` resume + atomic pickle |
| `process_item_blip2.py` | games SHA256 skip | skip / optional | n/a for MovieLens | — | — |

### 3.4 Schemas, validation, compatibility

| Official / need | Target file | Migrate how | Keep as-is? | Adaptations |
|-----------------|-------------|--------------|-------------|-------------|
| Implicit pickle schema | `schemas.py` | typed docs + validators | field set | TypedDict / dataclass for checks only |
| Downstream loader expectations | `compatibility.py` | new | official fields immutable | `load_official_compatible_dataset`, `validate_official_schema`, optional `convert_to_llm4rec_bias_schema` / lab schema without mutating `meta` / `meta_img_des` |
| Stats / reports | `validator.py` | new | — | `validation_report.json` + `.md` |
| Mode switch | `compatibility.py` + `config.py` | new | default `original` | `CompatibilityMode` enum |

### 3.5 Config / CLI / scripts / tests

| Need | Target | Notes |
|------|--------|-------|
| Args + YAML | `config.py`, `configs/dataset/mllm4rec_*.yaml` | replace hardcoded paths/device/key |
| Staged CLI | `cli.py` | `download`, `preprocess`, `match-tmdb`, `download-posters`, `generate-captions`, `serialize`, `validate`, `build` |
| Shell | `scripts/mllm4rec/*.sh` | thin wrappers calling module CLI |
| Constants | `constants.py` | folder name template, default model id, image base URL |
| Tests | `tests/data/mllm4rec/*` | mocks for TMDb/BLIP2; no real model download in unit tests |
| Attribution | file headers + docs | `# Adapted from: https://github.com/wangyuxiang123/MLLM4Rec` |

---

## 4. Target module responsibilities (checklist)

```text
src/llm4rec_bias_Integrated/data/mllm4rec/
├── __init__.py              # public: build/load helpers; no side effects
├── cli.py                   # argparse / subcommands
├── config.py                # MLLM4RecDataConfig from YAML + CLI + env
├── constants.py             # path templates, defaults
├── dataset_factory.py       # code → dataset class
├── base_dataset.py          # AbstractDataset port + path helpers
├── movielens_100k.py        # official ml-latest-small (+ classic optional)
├── movielens_1m.py          # extension
├── filtering.py             # filter_triplets (original iterative)
├── splitting.py             # split_df LOO
├── id_mapping.py            # densify + JSON maps + invertibility
├── tmdb_client.py           # original vs robust match
├── poster_downloader.py     # download + resume + failure log
├── blip2_captioner.py       # original generate semantics
├── serializer.py            # dataset.pkl (+ parquet sidecars, atomic/bak)
├── validator.py             # reports + schema asserts
├── compatibility.py         # official load/validate/convert
└── schemas.py               # OfficialDatasetDict field contract
```

---

## 5. Existing `llm4rec_bias_Integrated` reuse matrix

| Existing module | Reuse? | Why |
|-----------------|--------|-----|
| `datasets/movielens/ml100k.py` (classic `u.*`) | **No for original** | Different raw format and ID/schema vs MLLM4Rec |
| `datasets/movielens/preprocess.py` `MovieLensAdapterBase` | **No for pickle path** | Letter-route: rating threshold, string ids, candidate examples |
| `datasets/movielens/split.leave_one_out_split` | **Do not call for original** | Similar LOO idea but Interaction objects + different filters; keep MLLM4Rec `split_df` port separate |
| `datasets/movielens/common.chronological_sequences` | optional later for convert | Sort key may differ (`sid` vs row index) |
| `core/reproducibility.write_json` / fingerprint | **yes** for reports/sidecars | Engineering only |
| `tracking/logger` | **yes** | Replace print-heavy official scripts |
| `compatibility/llm4rec_bias_eval.py` | **do not mix** | Eval adapter ≠ MLLM4Rec data schema |
| `workflows/mllm4rec.yaml` | leave as letter stub for now | Multimodal train out of scope; data pipeline is separate CLI |

**Integration hook (later, non-breaking):**

```python
# suggested, Phase 3+
from llm4rec_bias_Integrated.data.mllm4rec.compatibility import load_official_compatible_dataset

def load_dataset(..., workflow="mllm4rec", compatibility_mode="original"):
    ...
```

Existing `build_dataset(...)` for letter/SID remains unchanged.

---

## 6. Pipeline stages ↔ CLI ↔ artifacts

| Stage | CLI | Primary outputs | Resume rule |
|-------|-----|-----------------|-------------|
| download | `download` | raw under `{raw_dir}` | skip if required files exist |
| preprocess | `preprocess` | `dataset.pkl` without `meta_img_des` | skip unless `--overwrite` |
| match-tmdb | `match-tmdb` | `tmdb_matches.jsonl` | skip cached ids |
| download-posters | `download-posters` | `img/{internal_id}.jpg`, `failed_posters.jsonl` | skip existing files unless `--overwrite` |
| generate-captions | `generate-captions` | `captions.jsonl`, update pickle `meta_img_des` | resume from jsonl |
| serialize | `serialize` | parquet/json sidecars + atomic pickle refresh | backup `.bak` when rewriting |
| validate | `validate` | `validation_report.json|.md` | always recompute |
| build | `build` | all stages in order | each stage respects resume |

---

## 7. original vs robust (locked defaults)

```yaml
compatibility:
  mode: original   # DEFAULT — must match official semantics

filtering:
  mode: original   # iterative min_uc/min_sc; min_rating NOT applied as filter

tmdb:
  match_mode: original  # title[:-7], first hit

caption:
  mode: original        # generate(**inputs) only; batch_size=1; processor-in-loop OK
```

| Behavior | original | robust |
|----------|----------|--------|
| Raw for `ml-100k` | ml-latest-small | same (classic is other code) |
| `min_rating` filter | unused (path only) | optional real rating filter if enabled |
| Sort tie-break | `timestamp`, `sid` | optional `source_row_index` |
| TMDb | first result | year / similarity / reject |
| Poster | no retry; skip on fail | retry, PIL verify, atomic |
| Caption | official call shape | batch + single processor load |
| Sidecars | optional | parquet/jsonl/reports |
| Pickle fields | official only | may add `extended_metadata` key only |

---

## 8. Official defects preserved in original (explicit)

1. `min_rating` does not filter ratings.  
2. TMDb always takes first search hit.  
3. Bare exception → empty URL / skip (target: catch `Exception`, same outcome; no silent `pass` without log in robust).  
4. No poster → keep item; caption `""`.  
5. BLIP2: no RGB convert; no generate kwargs; processor inside loop.  
6. Secondary sort key is `sid`, not row index.  
7. Dataset code name `ml-100k` means ml-latest-small.

Robust may fix (2)–(5) and optionally (6) **only** when mode ≠ original, with tests proving divergence.

---

## 9. Implementation order (Phases 3–9 reminder)

| Phase | Deliverable | Depends on this mapping |
|-------|-------------|-------------------------|
| 3 | preprocess → `dataset.pkl` for `ml-100k` (no TMDb/BLIP2) | §§3.1, 5, 7 |
| 4 | TMDb + posters, `--max-items 20` | §3.2 |
| 5 | BLIP2 captions, 20 items | §3.3 |
| 6 | compatibility tests vs loader field needs | §3.4 |
| 7 | full ml-100k + validation report | §6 |
| 8 | ml-1m extension | `movielens_1m.py` |
| 9 | tests, README pipeline doc, compatibility report | all |

---

## 10. Phase 2 report

```text
Phase: 2 — Migration mapping design
Official code inspected: YES (reused Phase 1 audit + process_item / blip2 / base / ml_100k)
Files created: docs/mllm4rec_data_migration_mapping.md
Files modified: (none in src/)
Functions migrated: (none — design only)
Behavior preserved: N/A (locked as design constraints for later phases)
Engineering changes: package path decided as llm4rec_bias_Integrated.data.mllm4rec;
  dataset codes: ml-100k=official ml-latest-small, ml-100k-classic & ml-1m=extensions;
  isolate from existing MovieLensAdapter letter-route
Commands executed: inspect repo tree + official sources
Tests passed: N/A
Tests failed: N/A
Known issues: task sketch path src/llm4rec/... remapped to src/llm4rec_bias_Integrated/...;
  classic u.data is NOT original ml-100k
Next action: Phase 3 — implement MovieLens-100k (ml-latest-small) parser,
  filtering, densify, LOO split, dataset.pkl serialization (no TMDb/BLIP2)
```
