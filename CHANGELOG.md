# Changelog

## 2026-07-31

### Added
- **`evaluation.sh`**：minionerec SID 独立评估入口。默认评 **SFT**；在脚本内填写 `RUN_DIRS`（可选 `CHECKPOINTS`）后执行。日志写入 `logs/evaluation.txt`。
- **README「评估」一节**：说明训练内嵌指标、独立重评、`analyze` 探针与产物路径。

### Changed
- **`llm4rec evaluate`（`src/llm4rec/cli/evaluate.py`）**：识别 minionerec / SID 路线后走全目录 beam + bias 指标（`pop_lift`、`delta_gap`、Gini、coverage、IPS、head/mid/tail 等），不再误用 letter 聚合。
  - `checkpoint_stage=sft|grpo`（独立脚本默认 `sft`；CLI 对 SID 默认优先 GRPO）
  - 可选 `adapter_path=` / `sft_adapter_path=`
  - 优先读取 `run_dir/resolved_config.yaml`，结果写入 `eval/{sft|grpo}_metrics.json`
