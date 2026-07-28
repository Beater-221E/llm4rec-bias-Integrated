#!/usr/bin/env bash
# Launch the full multi-GPU pipeline under conda env `bias`.
# Each start clears previous training logs under logs/, then writes a fresh run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

# --- conda bias ---
if [[ -f /opt/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/miniconda3/etc/profile.d/conda.sh
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate bias
fi

FORCE="${FORCE:-0}"
FOREGROUND="${FOREGROUND:-0}"
# If set, skip letter SFT and start GRPO from this adapter/checkpoint directory.
LETTER_INIT_CHECKPOINT="${LETTER_INIT_CHECKPOINT:-}"
if [[ -n "$LETTER_INIT_CHECKPOINT" && ! -f "$LETTER_INIT_CHECKPOINT/adapter_config.json" ]]; then
  echo "ERROR: LETTER_INIT_CHECKPOINT missing adapter_config.json: $LETTER_INIT_CHECKPOINT" >&2
  exit 1
fi
# If set, skip letter SFT and start GRPO from this adapter/checkpoint.
LETTER_INIT_CHECKPOINT="${LETTER_INIT_CHECKPOINT:-}"
if [[ -n "$LETTER_INIT_CHECKPOINT" && ! -f "$LETTER_INIT_CHECKPOINT/adapter_config.json" ]]; then
  echo "ERROR: LETTER_INIT_CHECKPOINT missing adapter_config.json: $LETTER_INIT_CHECKPOINT" >&2
  exit 1
fi

# Refuse to start (and wipe logs) while a previous full_multi is still alive,
# unless FORCE=1.
shopt -s nullglob
for pidfile in "$LOG_DIR"/full_multi_*.pid; do
  pid="$(tr -d '[:space:]' <"$pidfile" || true)"
  [[ -n "${pid:-}" ]] || continue
  if kill -0 "$pid" 2>/dev/null; then
    if [[ "$FORCE" != "1" ]]; then
      echo "ERROR: previous full_multi still running (pid=$pid, $pidfile)." >&2
      echo "Stop it first, or relaunch with FORCE=1 to kill and clear logs." >&2
      exit 1
    fi
    echo "FORCE=1: stopping previous full_multi pid=$pid"
    # Kill the launcher process group if possible; fall back to the pid tree.
    kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    sleep 2
    kill -9 -- -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
    # Also sweep leftover train/accelerate children from this project.
    pkill -f "$ROOT/logs/full_multi_" 2>/dev/null || true
    pkill -f 'llm4rec_bias_Integrated\.cli\.main train' 2>/dev/null || true
    sleep 1
  fi
done
shopt -u nullglob

# --- clear historical training logs ---
echo "Clearing previous logs under $LOG_DIR ..."
shopt -s nullglob
rm -f "$LOG_DIR"/full_multi_*.log \
      "$LOG_DIR"/full_multi_*.pid \
      "$LOG_DIR"/full_multi_*.sh \
      "$LOG_DIR"/full_multi_*.status \
      "$LOG_DIR"/run.log
rm -f "$LOG_DIR"/mllm4rec/*.log "$LOG_DIR"/mllm4rec/*.txt 2>/dev/null || true
shopt -u nullglob
mkdir -p "$LOG_DIR/mllm4rec"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_PREFIX="$LOG_DIR/full_multi_${TS}"
RUN_SH="${RUN_PREFIX}.sh"
RUN_LOG="${RUN_PREFIX}.log"
RUN_PID="${RUN_PREFIX}.pid"
RUN_STATUS="${RUN_PREFIX}.status"

cat >"$RUN_SH" <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export LLM4REC_COMPOSE="${LLM4REC_COMPOSE:-hardware=multi scale=full}"
# export TMDB_API_KEY="..."   # MLLM 需要时再打开

run_step() {
  local name="$1"; shift
  echo "===== STEP $name $(date -Is) ====="
  if "$@"; then
    echo "OK $name" | tee -a "$STATUS"
  else
    local ec=$?
    echo "FAIL $name exit=$ec" | tee -a "$STATUS"
    exit "$ec"
  fi
}

STATUS="${STATUS:?}"

run_step letter_prepare python -m llm4rec_bias_Integrated.cli.main prepare experiment=smoke_test dataset=movielens_100k

LETTER_EXTRA=()
if [[ -n "${LETTER_INIT_CHECKPOINT:-}" ]]; then
  echo "Resume letter from SFT checkpoint: $LETTER_INIT_CHECKPOINT"
  LETTER_EXTRA+=(
    "training.stages=[grpo,evaluate,analyze]"
    "init_checkpoint=${LETTER_INIT_CHECKPOINT}"
  )
fi
run_step letter_train python -m llm4rec_bias_Integrated.cli.main train experiment=smoke_grpo \
  "${LETTER_EXTRA[@]}"

run_step sid_prepare python -m llm4rec_bias_Integrated.cli.main prepare experiment=smoke_sid
run_step sid_train   python -m llm4rec_bias_Integrated.cli.main train experiment=smoke_sid

run_step mllm_build env CUDA_VISIBLE_DEVICES=0 python -m llm4rec_bias_Integrated.data.mllm4rec.cli build \
  --config configs/dataset/mllm4rec_ml100k.yaml
run_step mllm_retriever env CUDA_VISIBLE_DEVICES=0 python -m llm4rec_bias_Integrated.mllm4rec.cli train-retriever \
  --config configs/training/mllm4rec_retriever.yaml
run_step mllm_ranker env CUDA_VISIBLE_DEVICES=0 python -m llm4rec_bias_Integrated.mllm4rec.cli train-ranker \
  --config configs/training/mllm4rec_ranker.yaml \
  --retrieved-pkl experiments/lru/ml-100k/retrieved.pkl

echo "DONE full_multi $(date -Is)" | tee -a "$STATUS"
EOF
chmod +x "$RUN_SH"

echo "Python: $(command -v python)"
echo "Log:    $RUN_LOG"
echo "Status: $RUN_STATUS"
if [[ -n "$LETTER_INIT_CHECKPOINT" ]]; then
  echo "Resume: $LETTER_INIT_CHECKPOINT"
fi

if [[ "$FOREGROUND" == "1" ]]; then
  echo $$ >"$RUN_PID"
  STATUS="$RUN_STATUS" LETTER_INIT_CHECKPOINT="$LETTER_INIT_CHECKPOINT" \
    bash "$RUN_SH" 2>&1 | tee "$RUN_LOG"
else
  nohup env STATUS="$RUN_STATUS" LETTER_INIT_CHECKPOINT="$LETTER_INIT_CHECKPOINT" \
    PATH="${CONDA_PREFIX:+$CONDA_PREFIX/bin:}$PATH" \
    bash "$RUN_SH" >"$RUN_LOG" 2>&1 &
  echo $! >"$RUN_PID"
  echo "Started pid=$(cat "$RUN_PID")"
  echo "tail -f $RUN_LOG"
fi
