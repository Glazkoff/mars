#!/bin/bash
# M32: Gemma-4-31B sentence decomposition, 8 shards.
#SBATCH --job-name=m3-decompose
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/user/results/mars3/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
i="${SLURM_ARRAY_TASK_ID:-0}"; plan_receipt "$M3/m32/receipt_decompose_$i"
"$PLAN_PY" "$M3CODE/m32_decompose.py" --shard "$i/8" --out "$M3/m32/props_shard$i.jsonl"
plan_done "$M3/m32/receipt_decompose_$i"
