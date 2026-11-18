#!/bin/bash
# M31 recovery writer, 3 seeds.
#SBATCH --job-name=m3-writer
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/user/results/mars3/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
read -r -a SEEDS <<< "${M31_SEEDS:-20260906 20260907 20260908}"; SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
OUT="$M3/m31/writer_seed${SEED}"; mkdir -p "$OUT"; plan_receipt "$OUT"
"$PLAN_PY" "$M3CODE/m31_writer.py" --seed "$SEED" --out "$OUT"
plan_done "$OUT"
