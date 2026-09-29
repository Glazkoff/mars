#!/bin/bash
# M31 label-efficiency curve: tasks 0-8 tagger, 9-17 tagger_pre; frac = (task/3)%3 in {0.1,0.25,0.5}, seed = task%3.
#SBATCH --job-name=m3-curve
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-17
#SBATCH --output=/home/user/results/mars3/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
FRACS=(0.1 0.25 0.5); read -r -a SEEDS <<< "${M31_SEEDS:-20260906 20260907 20260908}"; NS=${#SEEDS[@]}
T="${SLURM_ARRAY_TASK_ID:-0}"; HALF=$((3 * NS)); ARM=$([ $((T / HALF)) = 0 ] && echo tagger || echo tagger_pre); R=$((T % HALF)); FRAC="${FRACS[$((R / NS))]}"; SEED="${SEEDS[$((R % NS))]}"
EXTRA=""; [ "$ARM" = tagger_pre ] && EXTRA="--pretrain-rows $M3/m31/pseudo_all.jsonl"
TAG="${ARM}[frac${FRAC}]"; OUT="$M3/m31/${ARM}_frac${FRAC}_seed${SEED}"; mkdir -p "$OUT"; plan_receipt "$OUT"
"$PLAN_PY" "$M3CODE/m31_tagger.py" --tag "$TAG" --seed "$SEED" --out "$OUT" --label-frac "$FRAC" $EXTRA
plan_done "$OUT"
