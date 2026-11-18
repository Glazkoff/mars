#!/bin/bash
# B18: score the human-labelled ACUs with the B2 checkpoints (arms x 5 seeds). Env: B18_TAG (checkpoint family prefix, e.g. xdom-rose_samsum_), B18_SPLIT.
#SBATCH --job-name=g-b18-mars2
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-9
#SBATCH --output=/home/user/results/mars2_gates/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
ARMS=(full_recovery direct_enabled); read -r -a SEEDS <<< "${B18_SEEDS:-20260906 20260907 20260908 20260909 20260910}"; NS=${#SEEDS[@]}
T="${SLURM_ARRAY_TASK_ID:-0}"; ARM="${ARMS[$((T / NS))]}"; SEED="${SEEDS[$((T % NS))]}"; TAG="${B18_TAG:-}"; SPLIT="${B18_SPLIT:-validation}"
CKPT="$MODELS_B2/${TAG}${ARM}_seed${SEED}/epoch1"; [ -d "$CKPT" ] || CKPT="$MODELS_B2/${TAG}${ARM}_seed${SEED}/epoch0"
[ -d "$CKPT" ] || { echo "no checkpoint ${TAG}${ARM} seed $SEED"; exit 3; }
OUT="$G/b18/mars2${TAG:+_$TAG}"; mkdir -p "$OUT"; plan_receipt "$OUT/receipt_${ARM}_${SEED}"
"$PLAN_PY" "$CODE/b18_score_mars2.py" --ckpt "$CKPT" --units "$G/b18/acu_units_${SPLIT}.jsonl" --out "$OUT/${ARM}_seed${SEED}_${SPLIT}.jsonl" --batch 8
plan_done "$OUT/receipt_${ARM}_${SEED}"
