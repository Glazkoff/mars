#!/bin/bash
# MARS-3 Stage 0 diagnostics (CPU).
#SBATCH --job-name=m3-stage0
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/user/results/mars3/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
export MARS2_DEVICE=cpu; mkdir -p /home/user/results/mars3/stage0
plan_receipt /home/user/results/mars3/stage0
"$PLAN_PY" /home/user/summ-mask-benchmark/research/mars3_20260916/code/stage0_diagnostics.py --out /home/user/results/mars3/stage0
plan_done /home/user/results/mars3/stage0
