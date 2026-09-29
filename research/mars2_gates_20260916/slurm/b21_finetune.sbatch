#!/bin/bash
# B21: supervised DeBERTa-large cross-encoder on the B1 labels, 3 seeds; scores m2 + ACU units. Usage: sbatch --dependency=afterok:<tasks> b21_finetune.sbatch
#SBATCH --job-name=g-b21-xenc
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/user/results/mars2_gates/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
read -r -a SEEDS <<< "${B21_SEEDS:-20260906 20260907 20260908}"; SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
OUT="$G/b21/xenc_seed${SEED}"; mkdir -p "$OUT"; plan_receipt "$OUT"
"$PLAN_PY" "$CODE/b21_finetune_xenc.py" --seed "$SEED" --out "$OUT" --eval-acu "$G/b18/acu_units_validation.jsonl" --scores-out "$G/b21/scores" ${B21_EXCLUDE:+--exclude-resource $B21_EXCLUDE}
plan_done "$OUT"
