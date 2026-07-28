#!/usr/bin/env bash
# Download / preprocess MovieLens via the unified CLI.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET="${1:-movielens_100k}"
echo "Preparing dataset=${DATASET}"
llm4rec-bias-Integrated prepare dataset="${DATASET}" experiment=smoke_test
