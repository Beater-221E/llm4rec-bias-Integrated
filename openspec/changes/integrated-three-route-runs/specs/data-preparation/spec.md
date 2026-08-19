## ADDED Requirements

### Requirement: prepare.sh step contract
System SHALL expose data prep as `prepare.sh` with selectable `STEPS`: `download`, `data`, `embed`, `sid`, `bm25`. Each step SHALL skip work when output already exists unless `FORCE=1`.

#### Scenario: Skip download when Amazon23 raw files exist
- **WHEN** `Industrial_and_Scientific.jsonl.gz` + `meta_Industrial_and_Scientific.jsonl.gz` already on disk (incl. symlink from `/scratch/esd4uq/data_2/raw/`) and `STEPS` includes `download` without `FORCE=1`
- **THEN** download step does not fetch those files again

#### Scenario: Rebuild SID in artifacts layout
- **WHEN** `STEPS` includes `sid` and no matching `artifacts/sid/<dataset>/<hash>/` exists
- **THEN** system builds Semantic IDs in that directory, does not import `/scratch/esd4uq/data_2/processed/Industrial_and_Scientific/sid/`

### Requirement: Four-file processed contract
Processed data SHALL be `interactions.jsonl`, `item_meta.json`, `popularity.json`, `stats.json` under `data/processed/amazon23/<category>/`.

#### Scenario: Missing four-file contract triggers data step
- **WHEN** data step runs and those four files absent at `data/processed/amazon23/Industrial_and_Scientific/`
- **THEN** system writes them from existing Amazon23 jsonl.gz files, does not treat MiniOneRec `train.jsonl`/`valid.jsonl`/`test.jsonl` as contract
