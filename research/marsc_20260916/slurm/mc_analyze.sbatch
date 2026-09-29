#!/bin/bash
# MARS-C E1 analysis. Env: MC_GLOB (scores glob), MC_TAG (analysis subdir).
#SBATCH --job-name=mc-analyze
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; export MCCODE="$PLAN_REPO/research/marsc_20260916/code"; mkdir -p "$MC/logs" "$MC/e0" "$MC/e1" "$MC/scores" "$MC/analysis"
OUT="$MC/analysis${MC_TAG:+_$MC_TAG}"; mkdir -p "$OUT"; plan_receipt "$OUT"
"$PLAN_PY" "$MCCODE/mc_analyze.py" --data-dir "$MC/e0" --scores-glob "${MC_GLOB:-$MC/scores/acu_mc-*_seed*.jsonl}" --out "$OUT"
plan_done "$OUT"
