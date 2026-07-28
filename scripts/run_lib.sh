#!/usr/bin/env bash
# Shared helpers for smoke / grpo4rec / minionerec / mllm4rec runners.
# Caller sets ROOT and LOG. First source truncates LOG (overwrite).

: "${ROOT:?ROOT must be set}"
: "${LOG:?LOG must be set}"

mkdir -p "$(dirname "$LOG")"
if [[ "${_LLM4REC_LOG_READY:-0}" != "1" ]]; then
  : >"$LOG"
  _LLM4REC_LOG_READY=1
fi

# Progress / status → real terminal only.
# Prefer a concrete /dev/pts/N path: accelerate/elastic workers often have no
# controlling TTY, so opening "/dev/tty" fails even when the parent shell has one.
_TTY="/dev/tty"
_PROGRESS_FD=""
_TTY_PATH=""
if [[ -c "$_TTY" && -w "$_TTY" ]]; then
  exec 3>"$_TTY"
  _PROGRESS_FD=3
  # Resolve real pts. readlink(/proc/.../fd/3) often stays "/dev/tty"; ttyname() does not.
  _TTY_PATH=""
  if command -v python >/dev/null 2>&1; then
    _TTY_PATH="$(python -c 'import os; print(os.ttyname(3))' 2>/dev/null || true)"
  fi
  if [[ -z "$_TTY_PATH" || "$_TTY_PATH" == "/dev/tty" || ! -c "$_TTY_PATH" ]]; then
    _TTY_PATH="$(tty 2>/dev/null || true)"
  fi
  if [[ -z "$_TTY_PATH" || "$_TTY_PATH" == "not a tty" || "$_TTY_PATH" == "/dev/tty" || ! -c "$_TTY_PATH" ]]; then
    _cand="$(readlink "/proc/$$/fd/3" 2>/dev/null || true)"
    if [[ -n "$_cand" && "$_cand" != "/dev/tty" && -c "$_cand" ]]; then
      _TTY_PATH="$_cand"
    else
      _TTY_PATH="$_TTY"
    fi
  fi
  unset _cand || true
else
  _TTY="/dev/stderr"
  _TTY_PATH="$_TTY"
fi

tty_status() {
  # Status on its own line (not shared with the progress bar).
  if [[ -n "$_PROGRESS_FD" ]]; then
    printf '%s\n' "$*" >&3
  else
    printf '%s\n' "$*" >"$_TTY" 2>/dev/null || printf '%s\n' "$*"
  fi
}
tty_endl() {
  if [[ -n "$_PROGRESS_FD" ]]; then
    printf '\r\033[K%s\n' "$*" >&3
  else
    printf '\r\033[K%s\n' "$*" >"$_TTY" 2>/dev/null || printf '%s\n' "$*"
  fi
}

# Narrative → log file only.
log() { printf '%s\n' "$*" >>"$LOG"; }

# One task: all process output → log; terminal keeps stage header + progress bar.
step() {
  local name="$1"; shift
  log ""
  log "===== $name $(date -Is) ====="
  log "+ $*"
  tty_status "==> $name"
  export LLM4REC_PROGRESS_DESC="$name"
  # Concrete pts path first — survives multi-GPU relaunch without controlling TTY.
  export LLM4REC_PROGRESS_TTY="$_TTY_PATH"
  if [[ -n "$_PROGRESS_FD" ]]; then
    export LLM4REC_PROGRESS_FD="$_PROGRESS_FD"
  else
    unset LLM4REC_PROGRESS_FD || true
  fi
  # stdout/stderr → log; keep fd 3 as the live TTY for Python progress bars.
  if "$@" >>"$LOG" 2>&1; then
    tty_endl "OK $name"
    log "OK $name"
  else
    local ec=$?
    tty_endl "FAIL $name — see $LOG"
    log "FAIL $name exit=$ec"
    exit "$ec"
  fi
}

activate_bias() {
  if [[ -f /opt/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /opt/miniconda3/etc/profile.d/conda.sh
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  fi
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate bias 2>/dev/null || true
  fi
}
