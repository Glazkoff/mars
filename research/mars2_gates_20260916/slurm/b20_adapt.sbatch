#!/bin/bash
# B20: adapt the cross-resource fold checkpoints on label-free pseudo rows; tasks 0-5 = held-out domain rows (adapt-<D>_),
# 6-11 = control-domain rows (adaptctl-<D>_); within each half: arm = full_recovery (0-2) | direct_enabled (3-5), seed 1-3.
#SBATCH --job-name=g-b20-adapt
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=/home/user/results/mars2_gates/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
D="${B20_DOMAIN:-rose_samsum}"; C="${B20_CONTROL:-rose_xsum}"
ARMS=(full_recovery direct_enabled); read -r -a SEEDS <<< "${B20_SEEDS:-20260906 20260907 20260908}"; NS=${#SEEDS[@]}
T="${SLURM_ARRAY_TASK_ID:-0}"; HALF=$((2 * NS)); MODE=$((T / HALF)); R=$((T % HALF)); ARM="${ARMS[$((R / NS))]}"; SEED="${SEEDS[$((R % NS))]}"
if [ "$MODE" = 0 ]; then TAG="adapt-${D}_"; ROWS="$G/b20/pseudo_${D}.jsonl"; else TAG="adaptctl-${D}_"; ROWS="$G/b20/pseudo_${C}.jsonl"; fi
CKPT="$MODELS_B2/xdom-${D}_${ARM}_seed${SEED}/epoch1"; [ -d "$CKPT" ] || CKPT="$MODELS_B2/xdom-${D}_${ARM}_seed${SEED}/epoch0"
[ -d "$CKPT" ] || { echo "no fold checkpoint xdom-${D}_${ARM} seed $SEED"; exit 3; }
[ -s "$ROWS" ] || { echo "no pseudo rows $ROWS"; exit 3; }
OUT="$MODELS_B2/${TAG}${ARM}_seed${SEED}"; mkdir -p "$OUT"; plan_receipt "$OUT"
"$PLAN_PY" "$CODE/b20_adapt.py" --ckpt "$CKPT" --rows "$ROWS" --out "$OUT" --steps "${B20_STEPS:-600}" --seed "$SEED"
plan_done "$OUT"
