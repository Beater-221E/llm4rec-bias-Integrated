## Context

The grpo4rec pipeline operates within the `llm4rec-bias-Integrated` monorepo. All three pipelines (grpo4rec, minionerec, mllm4rec) share a single `config.yaml`, with global defaults (paths, models, hardware, tracking) and per-workflow sections. The CLI (`src/llm4rec/cli/main.py`) is Hydra-based, resolving `workflow=grpo4rec` plus optional `scale=smoke|full` and `hardware=single|multi` overlays.

Preprocessing is separate from training: `prepare` downloads and preprocesses, `train` reuses existing processed data. The code structure follows `src/llm4rec/` with subpackages for data, training, evaluation, and analysis.

## Goals / Non-Goals

**Goals:**
- Single entry point (`python -m llm4rec.cli.main`) for all grpo4rec operations
- Config-driven: all hyperparameters in `config.yaml`, no hardcoded values
- Reproducible: fixed seed, deterministic splits from processed data
- Supports rapid iteration (smoke scale, ~1 minute) and full training

**Non-Goals:**
- Multi-modal features — that is mllm4rec's scope
- Semantic IDs — that is minionerec's scope
- Multi-node distributed training — only single-box single/multi GPU

## Decisions

### LoRA over full fine-tuning
**Decision**: Use LoRA (rank=16, alpha=32) rather than full fine-tuning.
**Rationale**: Qwen2.5-0.5B has ~500M params; LoRA reduces trainable params to ~1M, enabling single-GPU training with batch_size=1-4. Full fine-tuning would require 4× more VRAM with no proven accuracy gain for this task scale.
**Alternatives considered**: Full fine-tuning (rejected: too VRAM-intensive); prompt-tuning (rejected: less expressive for structured output tasks).

### GRPO over PPO
**Decision**: Use Group Relative Policy Optimization rather than standard PPO.
**Rationale**: GRPO eliminates the value function model, halving memory requirements. For candidate selection with a defined reward function (exact_match + format_validity), the value function adds little benefit.
**Alternatives considered**: PPO (rejected: requires value model, more VRAM); DPO (rejected: requires pairwise preference data we don't have).

### Qwen2.5-0.5B as base model
**Decision**: Use Qwen2.5-0.5B-Instruct as default, with larger variants (1.5B, 3B, 7B) configurable.
**Rationale**: 0.5B is the smallest model that produces coherent structured outputs, fitting in <4GB VRAM with LoRA. Larger models are available for scaling experiments.
**Alternatives considered**: Llama-3.2-1B (rejected: larger, slower); Gemma-2-2B (rejected: different tokenizer ecosystem).

### Config hierarchy: defaults → scale override
**Decision**: Nested YAML config where `scale=smoke|full` overlays specific fields on top of workflow defaults.
**Rationale**: Avoids duplicating configs; smoke changes only the fields that need reduction (limits, steps, rank). All other params (model, split strategy, metrics) inherit from defaults.

### Leave-one-out split
**Decision**: For each user, use the last interaction as test, second-to-last as validation.
**Rationale**: Standard for sequential recommendation evaluation; mirrors real-world "predict next item" task.

## Risks / Trade-offs

- **[Risk] GRPO instability with small batch sizes**: With batch_size=1, policy updates can be noisy. → Use gradient accumulation (8 steps for GRPO) to effectively increase batch size. Reward normalization via `advantage_normalization=group`.
- **[Risk] Smoke scale not representative**: 64 train / 32 eval users may not reflect full dataset behavior. → Smoke is explicitly for rapid iteration; full results required before drawing conclusions.
- **[Risk] LoRA rank=8 in smoke may underfit**: Lower rank reduces model capacity. → Smoke is for correctness testing, not performance measurement. Full scale uses rank=16.
