#!/bin/bash
# MARS-C G5 -- untouched-pool feasibility census (UniSumEval-v1.0, QAPyramid): clone, census, licence, overlap with our sources. CPU + network.
#SBATCH --job-name=mc-g5-pool
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"; G5="$MC/g5"; mkdir -p "$G5"; plan_receipt "$G5"
unset HF_HUB_OFFLINE
"$PLAN_PY" "$MCCODE/mc_pool_feasibility.py" --our-sources "$G/b18/acu_units_validation.jsonl" "$G/b18/acu_units_test.jsonl" "$MC/e0/docs_train.jsonl" --out "$G5"
plan_done "$G5"
