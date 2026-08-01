#!/usr/bin/env bash
# 将 llm4rec 大输出目录与 HuggingFace 缓存迁到 Rivanna scratch，并在 repo 内建 symlink。
# 用法：在 repo 根目录执行  bash scripts/rivanna_scratch_setup.sh
# 可重复运行（已 symlink 的目录会跳过）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

USER="${USER:?USER must be set}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}/llm4rec-bias-Integrated}"

mkdir -p "$SCRATCH_ROOT"/{runs,logs,data,experiments,reports,.cache/huggingface}

link_or_move() {
  local name="$1"
  local src="$ROOT/$name"
  local dst="$SCRATCH_ROOT/$name"
  mkdir -p "$dst"

  if [[ -L "$src" ]]; then
    echo "OK  $name already -> $(readlink -f "$src")"
    return 0
  fi

  if [[ -d "$src" ]]; then
    shopt -s dotglob nullglob
    for f in "$src"/*; do
      [[ -e "$f" ]] || continue
      echo "mv  $f -> $dst/"
      mv "$f" "$dst"/
    done
    shopt -u dotglob nullglob
    rm -rf "$src"
  fi

  ln -sfn "$dst" "$src"
  echo "link $name -> $dst"
}

for d in runs data experiments logs reports; do
  link_or_move "$d"
done

HF_SRC="${HF_HOME:-$HOME/.cache/huggingface}"
HF_DST="$SCRATCH_ROOT/.cache/huggingface"
mkdir -p "$HF_DST" "$HOME/.cache"

if [[ -d "$HF_SRC" && ! -L "$HF_SRC" ]]; then
  shopt -s dotglob nullglob
  for f in "$HF_SRC"/*; do
    [[ -e "$f" ]] || continue
    echo "mv  $f -> $HF_DST/"
    mv "$f" "$HF_DST"/
  done
  shopt -u dotglob nullglob
  rm -rf "$HF_SRC"
fi

if [[ ! -L "$HF_SRC" ]]; then
  ln -sfn "$HF_DST" "$HF_SRC"
fi
echo "link huggingface -> $HF_DST"

export HF_HOME="$HF_DST"
export TRANSFORMERS_CACHE="$HF_DST"
echo ""
echo "Scratch root: $SCRATCH_ROOT"
echo "Add to ~/.bashrc (optional):"
echo "  export HF_HOME=$HF_DST"
echo "  export TRANSFORMERS_CACHE=$HF_DST"
