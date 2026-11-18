#!/bin/bash
# Cross-encoder pipeline on every available inventory (validation). Run after m32 inventories exist.
#SBATCH --job-name=m3-pipeline
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/results/mars3/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
UNITS=""; for F in "$M3"/m32/units_*_validation.jsonl; do [ -s "$F" ] && UNITS="$UNITS $(basename "$F" _validation.jsonl)=$F"; done
[ -n "$UNITS" ] || { echo "no inventories"; exit 3; }
plan_receipt "$M3/m31/pipeline"; "$PLAN_PY" "$M3CODE/m31_score_pipeline_all.py" --units $UNITS; plan_done "$M3/m31/pipeline"
