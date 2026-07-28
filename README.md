# llm4rec-bias-Integrated

模块化 **LLM4Rec Research Framework**：在同一工程中独立维护 MiniOneRec / MLLM4Rec / GRPO4Rec 三条 workflow，共享 Dataset / Model / Trainer / Reward / Evaluation 插件。

上游参考：[llm4rec-bias](https://github.com/dragonfly90/llm4rec-bias)。  
架构说明：[重构计划](docs/refactor_plan.md) · [迁移指南](docs/migration_guide.md)。

## 环境

- Python ≥ 3.11，**需要 CUDA**
- 推荐 conda 环境 `bias`（V100 用 `float16`）

```bash
conda activate bias
cd llm4rec-bias-Integrated
pip install -r requirements.txt
pip install -e .
python -m llm4rec.cli.main validate experiment=smoke_test
```

MLLM 拉海报需要 `export TMDB_API_KEY=...`（勿写入仓库）。

## 一键运行

终端只显示进度；完整日志写入 `logs/*.txt`（每次覆盖）。

| 脚本 | 内容 | 默认 |
|------|------|------|
| `./smoke.sh` | 三条路线限步冒烟 | 单卡 `scale=smoke` |
| `./grpo4rec.sh` | Letter：prepare → SFT/GRPO → eval | 多卡 `scale=full` |
| `./minionerec.sh` | SID：同上 | 多卡 `scale=full` |
| `./mllm4rec.sh` | Retriever → Ranker | 单卡 CLI |

```bash
./smoke.sh
./grpo4rec.sh                 # 单卡：HARDWARE=single ./grpo4rec.sh
./minionerec.sh
./mllm4rec.sh
```

## 配置

统一 compose（Hydra 风格）：

```bash
export LLM4REC_COMPOSE="hardware=multi scale=full"   # 默认 single + smoke
export PYTHONPATH=src

python train.py workflow=grpo4rec dataset=ml1m model=qwen25_3b \
  reward=bias_aware evaluation=full_bias

python -m llm4rec.cli.main prepare experiment=smoke_test
python -m llm4rec.cli.main train experiment=smoke_grpo
```

| 开关 | 文件 | 作用 |
|------|------|------|
| `hardware=single\|multi` | `configs/hardware/` | GPU、NCCL、是否自动多卡启动 |
| `scale=smoke\|full` | `configs/scale/` | 数据/步数限制 |
| `experiment=…` | `configs/experiments/` | 任务、模型、训练阶段 |
| `workflow=…` | `configs/workflows/` | minionerec / mllm4rec / grpo4rec |
| `reward=…` | `configs/reward/` | 可组合 reward 插件 |
| `evaluation=…` | `configs/evaluation/` | ranking / full_bias 等 |

旧包名 `llm4rec_bias_Integrated.*` 仍可用（兼容 shim）。

## 三条路线

| 路线 | 实验 | 输出 |
|------|------|------|
| **grpo4rec**（Letter） | `smoke_grpo` | `runs/.../grpo4rec/` |
| **minionerec**（SID） | `smoke_sid` | `runs/.../minionerec/` |
| **mllm4rec** | Retriever / Ranker YAML | `experiments/lru/`、`experiments/ranker/` |

手动 MLLM 示例：

```bash
python -m llm4rec.workflows.mllm4rec.data.cli build \
  --config configs/dataset/mllm4rec_ml100k.yaml
python -m llm4rec.workflows.mllm4rec._stack.cli train-retriever \
  --config configs/training/mllm4rec_retriever.yaml
python -m llm4rec.workflows.mllm4rec._stack.cli train-ranker \
  --config configs/training/mllm4rec_ranker.yaml \
  --retrieved-pkl experiments/lru/ml-100k/retrieved.pkl
```

## 目录

```text
config.yaml                 # 全局默认
configs/                    # hardware / scale / experiments / reward / evaluation …
src/llm4rec/                # 研究框架（canonical）
src/llm4rec_bias_Integrated/  # 兼容 re-export
train.py                    # 统一入口
scripts/                    # 共享脚本与 MLLM 辅助
data/                       # raw | processed | preprocessed（不入库）
runs/                       # Letter / SID 输出（不入库）
experiments/                # MLLM 输出（不入库）
logs/                       # 运行日志（不入库）
```

## 评测

```bash
python -m llm4rec.cli.main evaluate run_dir=runs/.../
python -m llm4rec.cli.main evaluate run_dir=runs/.../ \
  evaluation.predictions_only=true          # 可不占 GPU
python -m llm4rec.cli.main analyze \
  experiment=smoke_probes run_dir=runs/.../
```

## 文档与测试

- [重构计划](docs/refactor_plan.md)
- [迁移指南](docs/migration_guide.md)
- [多模态数据流水线](docs/mllm4rec_data_pipeline.md)

```bash
python -m pytest tests/framework -q
python -m pytest tests/unit -q
python -m pytest tests/data/mllm4rec tests/mllm4rec -q
```
