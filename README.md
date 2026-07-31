# llm4rec-bias-Integrated

三条独立路线：**grpo4rec** / **minionerec** / **mllm4rec**。  
配置全在根目录 `config.yaml`：每条路线下再分训练阶段。

## 环境

```bash
conda activate bias
cd llm4rec-bias-Integrated
pip install -r requirements.txt && pip install -e .
export PYTHONPATH=src
python -m llm4rec.cli.main validate experiment=smoke_test
```

## 数据集怎么建

三条路线**不共用同一份预处理结果**，先建好再训。

### grpo4rec（Letter）

经典 GroupLens **MovieLens-100K**（`u.data` / `u.item`）。

```bash
python -m llm4rec.cli.main prepare experiment=smoke_test dataset=movielens_100k
```

会下载到 `data/raw/movielens_100k/`，预处理写入 `data/processed/movielens_100k/`（交互、划分、流行度等）。  
`train` 时若已有 processed，会复用。

### minionerec（SID）

与 grpo4rec **同一套** MovieLens-100K processed，再额外建 Semantic ID：

```bash
python -m llm4rec.cli.main prepare experiment=smoke_sid
```

在 `data/processed/movielens_100k/sid/` 生成 codebook / `sid_*.jsonl`。  
`train experiment=smoke_sid` 若发现 SID 缺失也会补建。

### mllm4rec（Retriever + Ranker）

官方兼容管线：配置里的 `ml-100k` = GroupLens **ml-latest-small**（不是上面的 classic 100K）。

```bash
# 仅交互/标题 → dataset.pkl（冒烟够用）
python -m llm4rec_bias_Integrated.data.mllm4rec.cli build \
  --config mllm4rec_ml100k --skip-multimodal

# 完整多模态（海报 + BLIP2 caption）需要 API
export TMDB_API_KEY=...
python -m llm4rec_bias_Integrated.data.mllm4rec.cli build --config mllm4rec_ml100k
```

产物：`data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/dataset.pkl`（及 img/captions 等）。

| 路线 | 语料 | 主要产物 |
|------|------|----------|
| grpo4rec | classic ML-100K | `data/processed/movielens_100k/` |
| minionerec | 同上 + SID | `.../sid/` |
| mllm4rec | ml-latest-small | `data/preprocessed/.../dataset.pkl` |

## 一键跑

**`./smoke.sh` 假定数据已就绪**（不会帮你下全量多模态）。跑之前至少要有：

- `data/processed/movielens_100k/`（Letter / SID）
- `data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/dataset.pkl`（MLLM；可用上面 `--skip-multimodal` 生成）

```bash
./smoke.sh                # 三条路线限步冒烟（单卡）
./grpo4rec.sh             # 单卡：HARDWARE=single ./grpo4rec.sh
./minionerec.sh
./mllm4rec.sh             # 无完整 pkl+caption 时需要 TMDB_API_KEY
./evaluation.sh           # SID 独立评估（先填 RUN_DIRS；默认 SFT）
```

## 配置怎么读

```text
config.yaml
├── grpo4rec/     sft / grpo / evaluate / analyze + smoke|full
├── minionerec/   prepare / sft / grpo / evaluate + smoke|full
└── mllm4rec/     data / retriever / ranker + smoke|full
```

```bash
python -m llm4rec.cli.main train experiment=smoke_grpo
python -m llm4rec.cli.main train experiment=smoke_sid hardware=multi scale=full
python -m llm4rec.cli.main train workflow=grpo4rec scale=smoke

python -m llm4rec_bias_Integrated.mllm4rec.cli train-retriever --config mllm4rec_retriever
python -m llm4rec_bias_Integrated.mllm4rec.cli train-ranker --config mllm4rec_ranker \
  --retrieved-pkl experiments/lru/ml-100k/retrieved.pkl
```

## 评估

训练结束后，指标会写在对应 `run_dir` 的 `eval/` 下。也可事后单独重评。

### 训练时自动写出（推荐先看这里）

| 路线 | 何时算 | 产物 |
|------|--------|------|
| **minionerec（SID）** | SFT / GRPO 阶段结束后 | `eval/sft_metrics.json`、`eval/grpo_metrics.json` |
| **grpo4rec（Letter）** | 同上 | 同上（HR/NDCG/MRR/`pop_lift` 等） |

minionerec 的 SID 指标还包括 bias / 多样性：`pop_lift@1`、`delta_gap`、`exposure_gini`、`coverage@K`、`hr_ips@K`、`ndcg_ips@K`、`hr@K_{head,mid,tail}`、`free_gen_valid_rate`。

`stages` 里的 `evaluate` **不会再算一遍**，只是把已有 `eval_sft` / `eval_grpo` 挂到 `summary.json`。  
train 里的 `analyze` 写的是 `eval/reward_hacking.json`（reward vs held-out），不是 letter 探针。

### minionerec：独立重评（`evaluation.sh`）

```bash
# 1. 编辑 evaluation.sh，填入 run 目录（CHECKPOINTS 可留空）
RUN_DIRS=(
  "runs/movielens_100k/minionerec/<exp>/seed_42/<timestamp>"
)

# 2. 默认评 SFT → eval/sft_metrics.json
./evaluation.sh

# 评 GRPO
CHECKPOINT_STAGE=grpo ./evaluation.sh
```

等价 CLI：

```bash
export PYTHONPATH=src
python -m llm4rec.cli.main evaluate \
  run_dir=runs/.../20... \
  experiment=smoke_sid \
  checkpoint_stage=sft
```

前提：`data/processed/<dataset>/sid/` 已存在（`prepare experiment=smoke_sid` 或训练时已建）。

> `smoke_sid` 只是 **冒烟实验名**（少量数据、几步训练），不是正式全量评测预设。正式结果请指向真实 `run_dir`。

### grpo4rec：Letter 重评与偏置探针

```bash
python -m llm4rec.cli.main evaluate run_dir=$RUN_DIR
# 仅 CPU 重聚合已有 predictions
python -m llm4rec.cli.main evaluate run_dir=$RUN_DIR evaluation.predictions_only=true

# popularity / position / framing / recency 探针（需 GPU + checkpoint）
python -m llm4rec.cli.main analyze run_dir=$RUN_DIR
python -m llm4rec.cli.main analyze run_dir=$RUN_DIR checkpoint_stage=grpo
```

探针结果在 `eval/probes/`。这套探针面向 **letter** 路线；SID 的 bias 以 `*_metrics.json` 里字段为主。

### 更新说明

见 [CHANGELOG.md](CHANGELOG.md)。

## 测试 / 清理

```bash
make test
make clean    # 清缓存与 runs（保留 data/raw）
```
