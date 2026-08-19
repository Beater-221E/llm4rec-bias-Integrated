## Context

grpo4rec pipeline runs in `llm4rec-bias-Integrated` monorepo. Three pipelines (grpo4rec, minionerec, mllm4rec) share one `config.yaml`: global defaults (paths, models, hardware, tracking) + per-workflow sections. CLI (`src/llm4rec/cli/main.py`) Hydra-based, resolves `workflow=grpo4rec` + optional `scale=smoke|full`, `hardware=single|multi` overlays.

Preprocessing separate from training: `prepare` downloads + preprocesses, `train` reuses existing processed data. Code: `src/llm4rec/` subpackages for data, training, evaluation, analysis.

## Goals / Non-Goals

**Goals:**
- Single entry point (`python -m llm4rec.cli.main`) for all grpo4rec ops
- Config-driven: all hyperparams in `config.yaml`, no hardcoded values
- Reproducible: fixed seed, deterministic splits from processed data
- Rapid iteration (smoke scale, ~1 min) + full training

**Non-Goals:**
- Multi-modal features — mllm4rec's scope
- Semantic IDs — minionerec's scope
- Multi-node distributed training — single-box single/multi GPU only

## Decisions

### LoRA over full fine-tuning
**Decision**: LoRA (rank=16, alpha=32) instead of full fine-tuning.
**Rationale**: Qwen2.5-0.5B ~500M params; LoRA cuts trainable to ~1M, enables single-GPU with batch_size=1-4. Full fine-tuning needs 4× VRAM, no proven accuracy gain at this scale.
**Alternatives considered**: Full fine-tuning (rejected: too VRAM-heavy); prompt-tuning (rejected: less expressive for structured output).

### GRPO over PPO
**Decision**: Group Relative Policy Optimization instead of standard PPO.
**Rationale**: GRPO drops value function model, halves memory. For candidate selection with defined reward (exact_match + format_validity), value function adds little.
**Alternatives considered**: PPO (rejected: needs value model, more VRAM); DPO (rejected: needs pairwise preference data we lack).

### Qwen2.5-0.5B as base model
**Decision**: Qwen2.5-0.5B-Instruct default; larger variants (1.5B, 3B, 7B) configurable.
**Rationale**: Smallest model producing coherent structured outputs, fits <4GB VRAM with LoRA. Larger available for scaling experiments.
**Alternatives considered**: Llama-3.2-1B (rejected: larger, slower); Gemma-2-2B (rejected: different tokenizer ecosystem).

### Config hierarchy: defaults → scale override
**Decision**: Nested YAML config; `scale=smoke|full` overlays specific fields on workflow defaults.
**Rationale**: No config duplication; smoke changes only reduction fields (limits, steps, rank). Rest (model, split strategy, metrics) inherits defaults.

### Leave-one-out split
**Decision**: Per user: last interaction test, second-to-last validation.
**Rationale**: Standard for sequential recommendation eval; mirrors real "predict next item" task.

## Risks / Trade-offs

- **[Risk] GRPO instability with small batch sizes**: batch_size=1 → noisy policy updates. → Gradient accumulation (8 steps for GRPO) raises effective batch. Reward normalization via `advantage_normalization=group`.
- **[Risk] Smoke scale not representative**: 64 train / 32 eval users may not reflect full dataset. → Smoke only for rapid iteration; full results before conclusions.
- **[Risk] LoRA rank=8 in smoke may underfit**: Lower rank cuts capacity. → Smoke tests correctness, not performance. Full scale uses rank=16.
