#!/bin/bash
# B18: materialise the human-labelled ACU unit set + judge tasks (validation and test; CPU).
#SBATCH --job-name=g-b18-tasks
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/user/results/mars2_gates/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
plan_receipt "$G/b18/receipt_tasks"
MARS2_DEVICE=cpu "$PLAN_PY" "$CODE/b18_build_tasks.py" --splits validation,test --out "$G/b18"
plan_done "$G/b18/receipt_tasks"
