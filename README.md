# llm4rec-bias-Integrated

研究一个问题：**LLM4Rec 用 RL 训练时产生的 shortcut，是不是表现为 bias 被放大？**

为了让这个问题有答案，三条互不相同的 LLM4Rec 路线被统一到同一个 backbone、
同一份数据、同一套 bias 指标上 —— 这样 bias 的差异才能归因到训练方式，而不是
"它们本来就不一样"。

本仓库提供两种实验模式（同一套 trainer，用配置切换默认值）：

| Mode | 目的 |
|---|---|
| **`reproduction`** | 尽量忠实于原论文/原仓库的 SID、超参与阶段编排 |
| **`integrated`** | 统一数据 / backbone /（可选）SFT / 评测 / bias 分析，比较路线算法差异 |

**不要假设每一条路线在任何配置下都与原文逐行等价。** 只有 `mode: reproduction`
且对应官方依赖可用时，才按 reproduction 契约对齐；`integrated` 刻意保留统一研究设定。

### MiniOneRec reproduction scope (intentional)

MiniOneRec reproduction mode reproduces MiniOneRec's SID, SFT objective,
tokenization/prompt, GRPO/reward, and optimization semantics on the project's
unified dataset/split protocol.

It is therefore an **algorithm/training-semantic reproduction** rather than a
byte-for-byte reproduction of MiniOneRec's original preprocessed data files.

```yaml
reproduction_scope:
  method: minionerec
  algorithm_semantics: reference
  data_protocol: integrated_unified
```

Do **not** expect a MiniOneRec upstream CSV preprocessing pipeline in this repo.

---

## 目录

- [三条路线：原文 & 我们的实现](#三条路线原文--我们的实现)
- [Reproduction Mode vs Integrated Research Mode](#reproduction-mode-vs-integrated-research-mode)
- [框架是怎么统一的](#框架是怎么统一的)
- [环境](#环境)
- [跑一个实验](#跑一个实验)
- [硬件兼容性](#硬件兼容性)
- [配置](#配置)
- [结果在哪](#结果在哪)
- [bias 指标](#bias-指标)
- [wandb](#wandb)
- [多卡与 DeepSpeed](#多卡与-deepspeed)
- [Semantic ID：静态产物与碰撞语义](#semantic-id静态产物与碰撞语义)
- [换数据集 / 换 backbone](#换数据集--换-backbone)
- [代码结构](#代码结构)
- [与原文的差异](#与原文的差异)
- [常见问题](#常见问题)
- [附录：冒烟测试](#附录冒烟测试)

---

## 三条路线：原文 & 我们的实现

### 路线 1 — MiniOneRec（Semantic ID 生成式检索）

> 原文：[AkaliKong/MiniOneRec](https://github.com/AkaliKong/MiniOneRec)

LLM 直接生成物品的 Semantic ID token 序列，约束 beam 解码保证每条 beam 都是
合法且唯一的物品。三阶段：**SID 构建 → SFT → GRPO**。

| 项 | 原文 | `mode: reproduction` | `mode: integrated` |
|---|---|---|---|
| SID | 冻结 text encoder + **官方 3 层 RQ-VAE**（无 PCA；layers=`[2048..64]`；`e_dim=32`；codebook=`[256,256,256]`）；token `<a_12><b_200><c_7>` | 官方 RQ-VAE 移植 + Sinkhorn 冲突消解 | 简化 RQ-VAE（可 PCA）或 RQ-Kmeans |
| 碰撞 | 原始碰撞 → 冲突组 Sinkhorn 重分配（最多 20 轮）；**不要求最终碰撞率恒为 0** | 同左；仅 `strict_unique` 才硬失败 | 同左；512 codebook 等为实验变体 |
| SFT | **全参**：`ConcatDataset(SidSFT, SidItemFeat, FusionSeqRec)`；lr=`3e-4`；epochs=`10`；linear + warmup_steps=`20`；eval/save=`0.05`；`load_best_model_at_end`；left padding；raw tokenization（不走 chat template） | 对齐上游目标/超参/tokenization；micro-batch 可按显存下调 | 模块化 `seqrec` + `title2sid`/`sid2title`（chat template） |
| RL | GRPO，`num_generations=16`、`beta=1e-3`、`lr=1e-5`、cosine warmup、`beam_search` + **`do_sample=True`**、`sync_ref_model` | 对齐 | 对齐（可改采样） |
| reward | `ranking` = rule + ndcg_rule（目标文本精确匹配；无 −1 invalid） | 对齐（`implementation: minionerec_reference`） | 对齐 |
| 解码 | 前缀树约束 beam | 对齐；SID 碰撞确定性解码 | 对齐 |
| 数据 | Amazon Reviews 2014/2018/2023 | Amazon23 | Amazon23 |
| Reference pin | — | `reference.commit=0c64b955…`（写入 manifest） | — |

### 路线 2 — Rec-R1（检索 query 生成）

> 原文：[linjc16/Rec-R1](https://github.com/linjc16/Rec-R1)

**本仓库实现是 Rec-R1-style / paper-faithful integrated reimplementation**（自定义 GRPO + HF generate + Python BM25 + 统一 RuntimeContext）。**不是**官方 `verl` + FSDP + vLLM 执行栈的逐行复现；若需要严格官方栈，应作为可选独立 backend，而不是阻塞当前研究框架。

LLM 不直接输出物品，而是生成一条**检索 query**，交给检索器执行，检索指标直接当
reward。**原文没有 SFT 阶段**，直接从 instruct 模型开始 RL。

> ⚠️ **原文用的是 GRPO，不是 PPO。** verl 的入口文件叫 `main_ppo.py`，容易被误读，
> 但训练脚本 `scripts/train/train_rec-amazon_c4_3b.sh` 里写的是
> `algorithm.adv_estimator=grpo`，没有 critic。

| 项 | 原文 | 我们（integrated reimplementation） |
|---|---|---|
| 算法 | GRPO（基于 verl），`rollout.n=12`、`temperature=0.6`、`top_p=0.95`、`kl_loss_coef=0.001`、`low_var_kl`、`lr=1e-6` | 数学对齐；执行栈为自定义 GRPO（不依赖 verl） |
| 输出格式 | `<think>…</think><answer>{"query": "…"}</answer>`，answer 必须是含 `query` 键的合法 JSON | 一致 |
| 检索 | Pyserini / Lucene 多字段 BM25 | **纯 Python BM25**（支持 `NOT "…" AND …` 布尔语法）；`kind: lucene` 可切回 |
| reward | format ±0.1/−2 + NDCG@1000（训练）/ @100（验证），检索异常 −2 | 逐行对齐 `verl/utils/reward_score/amazon_c4.py` |
| 数据 | ESCI、Amazon-C4、**Amazon Reviews 2023** | Amazon23（原文本来就用它） |

### 路线 3 — DPO4Rec（偏好推理 + 重排）

> 原文：[arXiv 2410.05939](https://arxiv.org/abs/2410.05939)（ICME 2025，北大）。**未开源，本框架从零复现。**

> ⚠️ **这是 re-ranking，不是生成式检索。** LLM 产出的是"用户偏好推理文本"，
> 由一个传统重排器消费。ranked list 从**候选集**排出，不是全库。

论文 Algorithm 1：

```
for iteration in 1..T:                      # 论文实测 T=2 最好，T=3 因过拟合退化
    对每个 prompt 采 N=10 份推理文本
    用 reranker 给每份打分（重排后的 NDCG@5）  # reranker 就是 reward model
    chosen = argmax(score),  rejected = argmin(score)
    用 (prompt, chosen, rejected) 训 DPO      # β=0.01
```

| 项 | 原文 | 我们 |
|---|---|---|
| reranker | DLCM / PRM / SetRank 三选一 | 三个都实现，默认 **PRM**（论文里增益最稳） |
| 知识注入 | 冻结 PLM 编码推理文本 → Knowledge Adaptor 降维 → 与 ID 表征拼接 | 一致（encoder 冻结，只训 adaptor） |
| DPO | β=0.01，AdamW lr=5e-5，grad accum=8，batch=2，epoch=3 | 一致 |
| 迭代 | T=2，且把最优推理文本回灌 reranker 再训（双向增益，§IV-C-2） | 一致 |
| backbone | Llama3.1-8B-Instruct（另测 Mistral-7B / Yi-6B） | 统一到我们的 backbone |
| 数据 | ML-1M / Amazon-Books / Amazon-Beauty | Amazon23（ML-1M 适配器也提供，见「换数据集」） |

---


### Runtime wiring (devices / precision / strategy / memory)

```bash
GPUS=auto bash run.sh          # use all visible GPUs
GPUS=0 bash run.sh             # single GPU
GPUS=0,2 bash run.sh           # explicit subset

# hardware.precision=auto  → BF16 (cc>=8) / FP16 AMP / FP32
# hardware.strategy=auto   → single | ddp | fsdp (wired via RuntimeContext.wrap_model)
# hardware.memory=auto     → preserve global_batch_size while tuning micro-batch
# optimization.compile     → torch.compile / Inductor (off by default in reproduction)
```

NCCL P2P/IB are **not** disabled by default. Troubleshooting: `LLM4REC_NCCL_COMPAT=1`.

## Reproduction Mode vs Integrated Research Mode

```bash
# Reproduction（MiniOneRec 官方 SID / 超参）
EXP=minionerec_reproduction_qwen05b GPUS=0 bash run.sh mode=reproduction \
  hardware.devices=auto hardware.precision=auto hardware.strategy=auto

# Integrated（统一研究设定）
EXP=minionerec_qwen05b_amazon GPUS=0,1,2,3 bash run.sh mode=integrated
EXP=recr1_qwen05b_amazon GPUS=0 bash run.sh mode=integrated
EXP=dpo4rec_qwen05b_amazon GPUS=0 bash run.sh mode=integrated
```

| | Reproduction | Integrated |
|---|---|---|
| MiniOneRec SID | 官方 RQ-VAE，**无 PCA**，codebook 256 | 简化 RQ-VAE（可 PCA）/ RQ-Kmeans |
| Rec-R1 stages | 默认不强制额外 SFT | 常含 SFT 以对齐三条路线基线 |
| DPO4Rec | 论文级超参；实现细节有假设处会注明 | 同上 + 统一 backbone/数据 |
| 碰撞率 = 0 | **不是**硬性前提 | 同左 |
| codebook 512 | 实验变体，非默认 | 实验变体 |

## 硬件兼容性

Runtime 用 PyTorch capability API 选精度/策略，**不按 GPU 商品名写死分支**。

| GPU | Preferred precision | Supported |
|---|---|---|
| V100 | FP16 AMP / FP32（MiniOneRec SID SFT 默认 FP32） | yes |
| A100 | BF16 | yes |
| H200 | BF16 | yes |
| B100 | BF16 | yes |

FP8 永不自动开启。吞吐量不保证跨硬件一致。多卡默认启用 NCCL 拓扑检测；排查问题时：

```bash
LLM4REC_NCCL_COMPAT=1 bash run.sh
```

`global_batch_size`（alias `target_global_batch_size`）会随 `world_size` 重算
`gradient_accumulation_steps`。默认 `hardware.batch_policy.preserve_global_batch: best_effort`：
允许小幅相对偏差并写入日志；`strict` 才会在偏差超限时失败。

### Reproduction vs hardware-equivalent execution

| Layer | Reproduction preserves | May adapt across GPUs |
|---|---|---|
| Algorithm | SID recipe, reward/loss math, GRPO/DPO, sampling temperature/top-p/beams, eval | — |
| Hardware | — | micro-batch, grad accum, precision, strategy (DDP/FSDP), activation checkpointing, compile, attention backend, pad_to_multiple_of |

Each run writes `execution_manifest.yaml` with **algorithm** knobs vs **actual_execution**
(resolved/effective strategy, batch deviation, compile status, peak VRAM).

### Feature honesty labels

| Feature | Status |
|---|---|
| Soft global-batch `best_effort` | **implemented** |
| MiniOneRec reproduction SFT (SidSFT+SidItemFeat+FusionSeqRec; raw tokenization; left pad) | **implemented** |
| GRPO true B×G scoring + within-prompt advantages | **implemented** |
| GRPO LR scheduler + warmup (optimizer steps) | **implemented** |
| Constrained beam `do_sample` (reproduction True) | **implemented** |
| DPO true 2B preference minibatch | **implemented** |
| `sync_ref_model` TR-DPO mixup (α=0.6 / every 512) | **implemented** |
| Strategy requested/resolved/effective + DeepSpeed honesty for custom loops | **implemented** |
| FSDP wrap → optimizer → full-state save; precision from RuntimeContext | **implemented** |
| `memory:auto` (GRPO + SFT, representative shapes) | **implemented** |
| Attention `sdpa` + eager fallback | **implemented** |
| Length bucketing | **not implemented** (use HF `group_by_length` if needed) |
| Static KV cache | **experimental** (Rec-R1 opt-in; MiniOneRec constrained gen stays dynamic; fallback updates effective) |
| SID token init | **reference** resize-only in reproduction; optional `mean_noise` for integrated |
| Multi-GPU validation | `scripts/validate_multi_gpu.sh` + `scripts/validate_runtime_matrix.py` |
| MiniOneRec reference reward (rule + ndcg_rule; target text; no −1 invalid) | **implemented** |
| Reference pin (`reference.commit` → `execution_manifest`) | **implemented** |
| Triton RQ distance+argmin | **optional / not justified** — see `benchmarks/bench_rq_quantization.py` |
| Rec-R1 official verl/vLLM stack | **not implemented** (integrated reimplementation only) |

## 框架是怎么统一的

三条路线的 LLM 输出**完全不同**，所以统一点不可能在 LLM 输出层：

```
MiniOneRec : LLM → SID token 序列 → 约束 beam 解码      ┐
Rec-R1     : LLM → 检索 query     → BM25 检索           ├→ ranked item list
DPO4Rec    : LLM → 偏好推理文本   → adaptor + reranker  ┘
```

**唯一可比的接缝是最终的 ranked item list。** 所以所有 bias 指标都算在这一层，
三条路线共用同一份实现。路线之间的差异被收敛成一个 `Decoder` 接口
（`src/llm4rec/decoders/`）—— 加新路线只要实现一个 Decoder，bias 评测、
wandb、stage 编排全部自动复用。

```
                        configs/  ← 只改这里
                            │
       run.sh ─→ cli/main.py ─→ pipeline.py（stage 编排）
                                    │
        ┌───────────────────────────┼──────────────────────────┐
        ▼                           ▼                          ▼
  data/（注册表）              trainers/                    eval/
  amazon23 / movielens         sft  (全参 + DeepSpeed)      catalog
        │                      grpo (DDP)                   bias   ★统一
        ▼                      dpo  (DDP)                   online (按 step)
   统一四件套契约
        │
        ▼
   sid/（RQ-VAE 静态产物，hash 校验）
        │
   ┌────┴─────────────┬────────────────────┐
   ▼                  ▼                    ▼
constrained_beam   bm25_query      knowledge_reranker
   └────────────┬───┴────────────────────┘
                ▼
        ranked item list  ← ★ bias 全在这层算
                ▼
        tracking/（wandb 单 run 贯穿 + jsonl 兜底）
```

---

## 环境

```bash
conda create -n bias python=3.11 -y && conda activate bias

# PyTorch 按你的 CUDA 装，例如 CUDA 12.1：
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt && pip install -e .

# 3B+ 想用 ZeRO 才需要：
pip install deepspeed

wandb login    # 不想用就在 run.sh 里设 WANDB_MODE=disabled
```

| 项 | 要求 |
|---|---|
| Python | ≥ 3.11 |
| GPU | NVIDIA。0.5B 全参 fp32：SFT ≈14 GB，RL ≈18 GB（policy + 冻结 ref + G=16 采样） |
| 精度 | V100 没有 bf16，且 fp16 下 SID SFT（新增 embedding）会发散 → 默认 **fp32**。A100/H100 改 `bf16`，省一半显存 |

**本框架不装也不用 `peft`。** 官方 MiniOneRec 的 SFT 就是全参；而且 LoRA 会低估
RL 对表征的改动，与后续的表征分析目标冲突。配置里出现 `peft` 会直接报错。

---

## 跑一个实验

### 第一步：看有什么能跑

```bash
python -m llm4rec.cli.main list
```

```
实验名                        route       backbone               数据集                              stages
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
minionerec_qwen05b_amazon  minionerec  Qwen2.5-0.5B-Instruct  amazon23/Industrial_and_Scientific  sft,eval,rl,eval
recr1_qwen05b_amazon       recr1       Qwen2.5-0.5B-Instruct  amazon23/Industrial_and_Scientific  sft,eval,rl,eval
dpo4rec_qwen05b_amazon     dpo4rec     Qwen2.5-0.5B-Instruct  amazon23/Industrial_and_Scientific  train_reranker,sft,eval,dpo,eval
minionerec_qwen7b_amazon   minionerec  Qwen2.5-7B-Instruct    amazon23/Industrial_and_Scientific  sft,eval,rl,eval
...
```

末尾会印出常用覆盖命令，忘了语法直接看那里。

### 第二步：准备数据（**只跑一次**）

```bash
EXP=minionerec_qwen05b_amazon bash prepare.sh
```

依次做四件事，都带缓存，重跑会跳过已完成的：

| 步骤 | 产物 | 说明 |
|---|---|---|
| `download` | `data/raw/` | 自动下载。Amazon23 Industrial = 644 MB 评论 + 268 MB 元数据，**带断点续传** |
| `data` | `data/processed/` | 过滤 → k-core → leave-one-out 划分 → 四件套契约 |
| `embed` | `artifacts/embeddings/` | 冻结 encoder 编码物品文本（需 GPU） |
| `sid` | `artifacts/sid/<hash>/` | RQ-VAE 量化 → Semantic ID（需 GPU，几十分钟） |
| `bm25` | `artifacts/bm25/` | 倒排索引（Rec-R1 路线用；纯 CPU） |

只跑某几步：`STEPS=download,data bash prepare.sh`
强制重建：`FORCE=1 STEPS=sid bash prepare.sh`

### 第三步：训练

```bash
EXP=minionerec_qwen05b_amazon GPUS=0,1,2,3 bash run.sh
```

就这一行。`GPUS` 是唯一需要按机器改的东西，多卡会自动起 torchrun。

跑之前会先做一次**纯 CPU 的配置校验**并打印执行计划 —— 不用等模型加载完
才发现参数配错：

```
实验          : minionerec_qwen05b_amazon  (route=minionerec)
阶段          : sft → eval → rl → eval
数据          : amazon23 / Industrial_and_Scientific
Backbone      : Qwen/Qwen2.5-0.5B-Instruct  精度=fp32  全参=True
解码器        : constrained_beam
随机种子      : 42
  sft         : lr=1e-05  epochs=3
  rl          : lr=1e-05  epochs=2  eval_steps=25  bias_eval_steps=25
bias 在线评测 : ['rl']（SFT 阶段按设计不评）
wandb         : online / llm4rec-bias
```

**SFT → 评测 → RL → 评测 是全自动串起来的**，不用人工接力。

### 临时改参数（不动文件）

```bash
bash run.sh train.rl.eval_steps=10 train.sft.learning_rate=2e-5
```

### 只评测已有 checkpoint

```bash
STAGES=eval RESUME_FROM=runs/.../rl/final bash run.sh
```

---

## 配置

分层组合。实验文件顶部用 `defaults:` 声明加载哪些层，逐层深度合并，
自己的键最后覆盖：

```
configs/
├── base.yaml                     全局：paths / hardware / wandb / checkpoint 策略
├── data/
│   ├── amazon23.yaml             Amazon Reviews 2023
│   └── movielens.yaml            ML-1M / ML-100K
├── model/
│   ├── qwen2.5-0.5b-instruct.yaml
│   ├── qwen2.5-3b-instruct.yaml
│   ├── qwen2.5-7b-instruct.yaml
│   └── llama3.1-8b-instruct.yaml
├── sid/rqvae.yaml                RQ-VAE 参数 + SID 导入
├── bias/default.yaml             bias 指标集合
├── deepspeed/                    zero2 / zero2_offload / zero3（YAML，不是 JSON）
├── route/minionerec/             MiniOneRec 共用层（common / integrated / reproduction）
└── exp/                          ★ 实验入口，跑实验的人只看这里
    ├── minionerec_qwen05b_amazon.yaml
    ├── recr1_qwen05b_amazon.yaml
    ├── dpo4rec_qwen05b_amazon.yaml
    ├── *_qwen7b_amazon.yaml             ← 薄覆盖，只写和 0.5B 不同的部分
    └── smoke_*.yaml                     ← 见附录
```

Rec-R1 / DPO4Rec 的 `*_qwen05b_amazon.yaml` 是自包含的（超参摊开在 exp 文件里）。
MiniOneRec 会先加载 `configs/route/minionerec/*`，再被 exp 文件覆盖。
想调什么直接搜对应 yaml。`EXP=` 必须是 `configs/exp/<name>.yaml` 的文件名
（例如 `minionerec_qwen05b_amazon`），没有 `minionerec.yaml` 这种短名。

### RL 按 step 评测

RL 是按 step 跑的（不是按 epoch），所以评测频率也按 step 配：

```yaml
train:
  rl:
    max_steps: null          # 非 null 时优先于 epochs
    eval_steps: 25           # 每 25 step 评一次准确率
    bias_eval_steps: 25      # 每 25 step 评一次 bias（null = 跟随 eval_steps）
    eval_examples: 256       # 在线评测的固定 held-out 子集大小
```

**在线评测不落 checkpoint。** 0.5B 全参一份约 2 GB，按 step 存会迅速吃满盘。
做法是每 N 步对一个固定 seed 采样的子集跑一次解码，当场算 bias 直接推 wandb。
子集全程不变，所以跨 step 的曲线是可比的。

---

## 结果在哪

```
runs/<数据集>/<route>/<模型>/seed_<seed>/<时间戳>/
├── resolved_config.json    这次跑用的完整配置 —— 拿它就能复现
├── execution_manifest.yaml algorithm vs actual_execution（batch/strategy/compile/VRAM）
├── environment.json        git commit / 依赖版本 / 硬件
├── metrics.jsonl           全部指标（wandb 挂了也不丢）
├── summary.json            各 stage 的 checkpoint 路径 + 最终指标
├── sft/
│   ├── final/              SFT 完的全参权重（可直接 RESUME_FROM）
│   └── train_log.json
├── rl/                     MiniOneRec / Rec-R1
│   ├── final/
│   └── train_log.json
├── reranker/               DPO4Rec：第一阶段 reranker
├── dpo/                    DPO4Rec：迭代 DPO
│   └── final/
└── eval/
    ├── eval_1.json         SFT 后的 bias 基线
    ├── eval_2.json         RL / DPO 后的 bias
    └── bias_delta.json     ★ 两者差值 = "RL 放大了多少 bias"
```

`bias_delta.json` 会自动生成，同时推到 wandb summary —— 多个 run 在 wandb
表格里可以直接按某个 delta 排序横向对比。

---

## bias 指标

所有指标都算在**最终的 ranked item list** 上，三条路线共用。
数学定义复用 [llm4rec-bias](https://github.com/dragonfly90/llm4rec-bias) 移植过来的
纯函数（`compatibility/llm4rec_bias_eval.py`），数字位级一致。

| 类别 | 指标 | 含义 | RL 放大 bias 时的方向 |
|---|---|---|---|
| 流行度 | `pop_lift@1` / `pop_lift@K` | 推荐物品流行度分位 − 全库均值 | ↑ |
| | `delta_gap` | 推荐流行度 − **该用户自己历史**的流行度 | ↑ |
| 曝光集中 | `exposure_gini` | 全库曝光次数的 Gini | ↑ |
| | `exposure_entropy` | 曝光分布归一化熵 | ↓ |
| | `coverage@K` | topK 覆盖到的物品占全库比例 | ↓ |
| 长尾代价 | `hr@K_head/mid/tail` | 按目标物品流行度分层的命中率 | tail ↓ |
| | `tier_gap` | `hr_head − hr_tail` | ↑ |
| 去偏准确率 | `hr_ips@K` / `ndcg_ips@K` | 流行度加权 IPS 去偏后的准确率 | **↓ 而 `hr@K` ↑ = 铁证** |
| shortcut | `history_copy_rate` | 推荐结果里已在用户历史中的比例 | ↑ |
| | `top1_concentration` | 所有用户的 top1 落在多少个不同物品上 | ↑（个性化坍塌） |
| 健康度 | `valid_rate` | 模型输出的合法率 | — |

**为什么 `delta_gap` 比 `pop_lift` 更能说明问题**：它剔除了"这个用户本来就爱看
热门"的成分，衡量的是模型相对用户自身偏好的流行度漂移。

**为什么 IPS 是关键**：如果 `hr@K` 涨但 `hr_ips@K` 跌，说明准确率的提升主要来自
"猜热门"而不是真的学会了推荐。这正是我们要抓的 shortcut。

**流行度先验只统计 train split** —— 混进 test 的话，"模型偏向热门"和"热门本来
就更容易命中"就分不开了。

**按研究设计，SFT 阶段不做 bias 在线评测**（只在 stage 结束评一次基线），
bias 的漂移观测放在 RL/DPO 阶段。

> DPO4Rec 是 re-ranking，ranked list 从候选集排出而不是全库，所以它的
> `coverage` / `exposure_gini` 的分母和另两条路线不同，跨路线看这两个指标时要留意。
> `pop_lift` / `delta_gap` / `tier_hr` / IPS 不受影响，仍然可比。

---

## wandb

**一个 run 贯穿所有 stage**，global step 连续，所以在 loss / reward 曲线上
能直接看出 RL 从哪一步接上，不用去对两个 run 的时间轴。

```
progress/stage_id          阶段指示线（sft=0, rl/dpo=1, eval=2）
train/loss                 训练损失
train/reward               GRPO 的平均 reward（多卡下是全局均值）
train/reward_std           组内 reward 方差 —— 趋近 0 说明 advantage 消失了
train/kl                   相对 reference model 的 KL
train/dpo_margin           DPO 的偏好间隔
train/accuracy             DPO 偏好方向对了的比例（最重要的健康指标）
bias/pop_lift@1            ★ 训练中每 N step 一个点
bias/exposure_gini
bias/tier_gap
bias/history_copy_rate
eval/hr@10  eval/ndcg@10   stage 结束的完整评测
```

bias 指标用 `define_metric` 声明为稀疏序列，不会被画成断线。

**wandb 挂了 / 没装 / 没登录都不影响训练** —— 所有调用都包了容错，
`run_dir/metrics.jsonl` 永远兜底。

关掉：`WANDB_MODE=disabled bash run.sh`；离线：`WANDB_MODE=offline`。

---

## 多卡与 DeepSpeed

```bash
GPUS=0,1,2,3 bash run.sh                    # DDP
DEEPSPEED=zero2 GPUS=0,1,2,3 bash run.sh    # SFT 用 ZeRO-2
```

| Stage | 并行方式 | 说明 |
|---|---|---|
| SFT | HF Trainer + DDP 或 **DeepSpeed ZeRO** | 由 `DEEPSPEED` 决定 |
| RL / DPO | **DDP** | 按 rank 分样本 + all-reduce 梯度；只有 rank0 写日志/存盘 |
| eval | 各 rank 解码一片 + all-gather 汇总 | 不让 rank0 独跑（其它 rank 干等会像卡死）|

⚠️ **DeepSpeed 只作用于 SFT。** RL 阶段刻意走 DDP：ZeRO-3 下参数被切分，
每次 `generate` 都要 gather 全部参数，而 GRPO 每个 step 要采 `group_size` 条，
会慢到不可用。

| 模型 | 建议 |
|---|---|
| 0.5B | `DEEPSPEED=`（留空，用 DDP）。单卡放得下，DDP 更快、更少坑 |
| 3B | `zero2`；显存紧张用 `zero2_offload` |
| 7B+ | `zero3` + bf16。注意 RL 阶段仍走 DDP，需要单卡能放下 policy + ref = 2 份模型 |

---

## Semantic ID：静态产物与碰撞语义

### 碰撞语义（MiniOneRec）

官方流程：

```text
initial RQ-VAE encoding
  → detect collision groups
  → re-encode only conflicting groups (last-level Sinkhorn)
  → retry up to 20 iterations
  → save final SID mapping
```

显式指标：`raw_collision_rate`、`post_resolution_collision_rate`、
`num_collision_groups`、`max_collision_group_size`、
`duplicate_item_collision_rate`、`quantization_collision_rate`。

- 原始碰撞会被测量并写入 `build_stats.json`
- 会做冲突消解，但**最终碰撞率不必为 0**（目录重复商品等）
- `sid.strict_unique: true` 才会因残留碰撞硬失败
- codebook `[512,512,512]` / constrained RQ-kmeans 是**实验配置**，不是 reproduction 默认修复
- Triton RQ `distance+argmin`：短 profile / `benchmarks/bench_rq_quantization.py` 显示
  开销相对 RQ-VAE 训练 / LLM forward **不构成热点** → 保持 `backend=reference`
  （placeholder 不是真 Triton；**custom Triton not justified by profiling**）

反向映射使用 `sid_to_items: dict[SID, list[ItemId]]`，评测需要唯一 item 时做确定性解析并记录歧义，绝不静默覆盖。



SID 是**静态产物**，只由 `prepare.sh` 生成一次，训练与评测全程**只读**：

```
artifacts/sid/<数据集>/<config_hash>/
├── item2sid.json     item → 码 + SID 字符串
├── codebook.pt       RQ-VAE 码本
├── build_stats.json  碰撞率、逐层码本利用率、PCA 保留方差
└── manifest.json     生成配置 + 物品集合指纹
```

训练启动时重算 hash 比对，对不上**直接报错退出，绝不隐式重建** —— 因为 SID 一变
物品的语义前缀全变，bias 指标（尤其 exposure / coverage）就不可比了，而这类问题
极难发现。改任何影响 SID 的参数都会落到新的 hash 目录，老产物原样保留。

### 导入现成的 SID

RQ-VAE 要 GPU、要几十分钟。在一台机器上建好，其它机器直接导入：

```bash
# 建好后整个目录拷走
scp -r artifacts/sid/amazon23_Industrial_and_Scientific/7b9e2a  训练机:/data/shared/sid/

# 训练机上直接指过去
bash run.sh sid.import_from=/data/shared/sid/7b9e2a
```

或写进配置：`sid: {import_from: /data/shared/sid/7b9e2a}`。
导入的产物**照样校验物品指纹**，对不上会报错，不会静默用错表。

找不到 SID 时的报错会把该数据集下已有的产物列出来，直接告诉你能导入哪个。

---

## 换数据集 / 换 backbone

都是纯配置，代码零改动：

```bash
# 换数据集
bash run.sh data.name=movielens data.variant=ml-1m

# 换 backbone
bash run.sh model.checkpoint=Qwen/Qwen2.5-3B-Instruct

# 两个都换
bash run.sh data.name=movielens data.variant=ml-1m model.checkpoint=Qwen/Qwen2.5-3B-Instruct
```

或改实验配置的 `defaults`：

```yaml
defaults:
  - base
  - data/movielens            # 换这行
  - model/llama3.1-8b-instruct  # 和这行
  - sid/rqvae
  - bias/default
```

### 加一个新数据集

1. 在 `src/llm4rec/data/` 下实现 `DatasetAdapter`，加 `@register_dataset`：
   - `raw_files()` 返回 `{说明: (下载URL, 本地落点)}` —— 下载和报错信息共用这一处
   - `preprocess()` 产出统一四件套契约
2. 写一个 `configs/data/<name>.yaml`

SID 构建、BM25、样本构建、bias 评测**全部不用改** —— 它们只认契约。
`movielens.py` 就是照这个流程加的，可以直接参考。

---

## 代码结构

```
src/llm4rec/
├── core/                 配置组合、分布式、异常、复现性
├── data/                 DatasetAdapter + 三条路线的样本构建
├── sid/                  SID 产物、RQ-VAE（integrated + MiniOneRec 官方）
├── decoders/             ★ 路线差异只在这里（SID beam / BM25 / reranker）
├── trainers/             SFT / GRPO / DPO / rollouts / rewards
├── runtime/              精度、策略、batch、显存探测、checkpoint、KV cache
├── retrieval/            纯 Python BM25
├── rerankers/            PRM / SetRank / DLCM + KnowledgeAdaptor
├── eval/                 统一 bias 指标 + 在线评测 + 多卡 gather
├── tracking/             console + jsonl + wandb + 进度条
├── compatibility/        上游 llm4rec-bias 指标桥
├── kernels/              可选 Triton RQ distance（默认关）
├── pipeline.py           stage 编排（SFT→eval→RL/DPO→eval）
└── cli/main.py           CLI
```

根目录 `workflow.ipynb` 是 Colab 安装附属，不是训练入口。

---

## 与原文的差异

全部是有意为之，理由如下：

| 差异 | 为什么 |
|---|---|
| **移除 LoRA，全参微调** | 官方 MiniOneRec `sft.py` 里本来就没有 peft；且 LoRA 会低估 RL 对表征的改动，与后续表征分析冲突。配置里出现 `peft` 会直接报错 |
| **Rec-R1 / DPO4Rec 加了 SFT** | 原文都没有。加它是为了让三条路线有**同一种基线** —— 否则 bias 变化里混着 instruct 模型自带的先验，归因不干净。要严格复现原文就把 `stages` 里的 `sft` 去掉 |
| **Rec-R1 用纯 Python BM25** | 原文是 Pyserini/Lucene。Amazon 单类目 ~1-2 万 item 量级足够，省掉 Java 依赖。`decoder.retriever.kind: lucene` 可切回 |
| **三条路线统一 Amazon23 + 同一 backbone** | 原文分别是 Amazon18/23、ESCI+Amazon-C4、ML-1M/Amazon-Books/Beauty；backbone 分别是 Qwen2.5 系、Qwen2.5-3B、Llama3.1-8B。统一之后数字**不与各自论文直接可比**，比的是三条路线之间 |
| **reward 里不加 bias 惩罚项** | 要观测的正是"纯准确率 reward 会不会放大 bias"。加了去偏项就把要测的现象抹掉了。做 mitigation 消融时另开配置 |

---

## 常见问题

**SID 碰撞率下不去 / 报错退出**
`mor-reproduce` 实测在 Amazon23 Industrial 上 RQ-VAE 的碰撞率高于 residual-kmeans，
它自己因此默认切到了 rqkmeans。我们按官方设定默认 `rqvae`，但设了
`sid.max_collision_rate: 0.0` 硬拦截 —— 不会带着脏 SID 训练。跑不下去就
`sid.method: rqkmeans`，或调大 `sid.codebook_size`。

**fp16 下 SFT loss 变 NaN**
V100 上新增 embedding 在 fp16 下会发散，这是已知问题。用 `fp32`（默认），
或换 A100/H100 用 `bf16`。

**多卡启动就卡住不动**
多半是 NCCL。默认**不**关 P2P/IB（让拓扑自检）。仅在 pre-Ampere GPU
或设置 `LLM4REC_NCCL_COMPAT=1` 时才会 disable P2P/IB。如果还卡，试
`NCCL_DEBUG=INFO` 看卡在哪一步。

**`train/reward_std` 一直是 0**
组内 reward 全一样 → advantage 全 0 → 这一组不产生梯度。GRPO 里这是正常现象
（全对或全错时），但如果**一直**是 0，说明任务对当前模型太难或太简单，
需要调 `group_size` 或先把 SFT 训好。

**显存不够**
按顺序试：`per_device_batch_size` 调小 → `gradient_accumulation_steps` 调大补回
全局 batch → RL 的 `grpo.group_size` 调小 → 上 `DEEPSPEED=zero2`（只帮 SFT）→
`zero2_offload`。

---

## 附录：冒烟测试

> 这一节是给 tester / debugger 的。正常跑实验用不到。

`configs/exp/smoke_*.yaml` 是几分钟就跑完的最小配置，**指标没有意义**，
只用来验证三件事：能不能跑通、checkpoint 是不是只有一份、wandb 有没有数。

新机器 / 新环境上建议按这个顺序验：

```bash
# 1. 单卡通不通
EXP=smoke_minionerec GPUS=0 bash run.sh

# 2. 多卡 DDP —— 最关键的一步
#    重点检查：loss 合理、checkpoint 只有一份、wandb 只有一个 run
EXP=smoke_minionerec GPUS=0,1 bash run.sh

# 3. DeepSpeed
DEEPSPEED=zero2 EXP=smoke_minionerec GPUS=0,1 bash run.sh

# 4. 另外两条路线
EXP=smoke_recr1   GPUS=0,1 bash run.sh
EXP=smoke_dpo4rec GPUS=0,1 bash run.sh
```

第 2 步是关键 —— RL/DPO 的分布式是手写的（不像 SFT 有 HF Trainer 兜底），
有问题会在那里暴露。
