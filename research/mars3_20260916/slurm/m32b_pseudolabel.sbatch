#!/bin/bash
# M32b: fact-level training rows (LLM inventory of the train pairs, cross-encoder teacher). Exits 3 if no LLM inventory reaches the ceiling bar.
#SBATCH --job-name=m3-m32b-label
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/results/mars3/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
plan_receipt "$M3/m32/receipt_m32b"
MARS2_DEVICE=cpu "$PLAN_PY" "$M3CODE/m32_pseudolabel.py" --props "$M3"/m32/props_shard*.jsonl --variant "${M32B_VARIANT:-auto}" --min-ceiling "${M32B_MIN_CEILING:-0.85}"
plan_done "$M3/m32/receipt_m32b"
