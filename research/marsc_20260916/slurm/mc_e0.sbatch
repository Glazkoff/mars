#!/bin/bash
# MARS-C E0: data gate (human TRAIN facts, clusters, splits, tetrad census, token-mask collisions). CPU.
#SBATCH --job-name=mc-e0
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; export MCCODE="$PLAN_REPO/research/marsc_20260916/code"; mkdir -p "$MC/logs" "$MC/e0" "$MC/e1" "$MC/scores" "$MC/analysis"
plan_receipt "$MC/e0"
"$PLAN_PY" "$MCCODE/mc_data.py" --out "$MC/e0"
plan_done "$MC/e0"
