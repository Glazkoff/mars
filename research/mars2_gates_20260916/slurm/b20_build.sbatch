#!/bin/bash
# B20: label-free pseudo rows for the held-out resource and for the control resource (CPU). Env: B20_DOMAIN, B20_CONTROL.
#SBATCH --job-name=g-b20-build
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user/results/mars2_gates/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
plan_receipt "$G/b20/receipt_build"
for D in "${B20_DOMAIN:-rose_samsum}" "${B20_CONTROL:-rose_xsum}"; do
  MARS2_DEVICE=cpu "$PLAN_PY" "$CODE/b20_build_pseudo.py" --domain "$D" --split train --n-cands 4 --out "$G/b20/pseudo_${D}.jsonl"
done
plan_done "$G/b20/receipt_build"
