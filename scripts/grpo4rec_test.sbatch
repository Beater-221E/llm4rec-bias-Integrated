#!/bin/bash
# grpo4rec task 7.3：单元测试（CPU）
#SBATCH -J grpo4rec_test
#SBATCH -A aikyamlab
#SBATCH -p standard
#SBATCH --time=0:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/grpo4rec_test_%j.out
#SBATCH --error=logs/grpo4rec_test_%j.err

set -euo pipefail
cd ~/llm4rec-bias-Integrated
mkdir -p logs
export PATH="$HOME/.conda/envs/bias/bin:$PATH"
export PYTHONPATH=src

# langsmith（pytest 间接依赖）需要 zstandard
python -m pip install -q zstandard

make test
echo "make test DONE"
