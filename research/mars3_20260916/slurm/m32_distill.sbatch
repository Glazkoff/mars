#!/bin/bash
# M32: distil the decomposition into LongT5-base and build the distilled inventory (GPU).
#SBATCH --job-name=m3-distill
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/results/mars3/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
plan_receipt "$M3/m32/receipt_distill"
"$PLAN_PY" "$M3CODE/m32_distill.py" --props "$M3"/m32/props_shard*.jsonl
for S in validation test; do MARS2_DEVICE=cpu "$PLAN_PY" "$M3CODE/m32_inventory.py" --split "$S" --variant distilled --props "$M3/m32/distilled_props.jsonl"; done
plan_done "$M3/m32/receipt_distill"
