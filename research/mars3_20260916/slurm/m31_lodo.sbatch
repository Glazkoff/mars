#!/bin/bash
# M31 LODO: tasks 0-11 tagger, 12-23 tagger_pre; within each half domain = (task/3)%4, seed = task%3.
#SBATCH --job-name=m3-lodo
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-23
#SBATCH --output=/home/user/results/mars3/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
DOMS=(rose_cnndm rose_xsum rose_samsum natural770); read -r -a SEEDS <<< "${M31_SEEDS:-20260906 20260907 20260908}"; NS=${#SEEDS[@]}
T="${SLURM_ARRAY_TASK_ID:-0}"; HALF=$((4 * NS)); ARM=$([ $((T / HALF)) = 0 ] && echo tagger || echo tagger_pre); R=$((T % HALF)); D="${DOMS[$((R / NS))]}"; SEED="${SEEDS[$((R % NS))]}"
EXTRA=""; [ "$ARM" = tagger_pre ] && EXTRA="--pretrain-rows $M3/m31/pseudo_all.jsonl"
TAG="${ARM}[xdom-${D}]"; OUT="$M3/m31/${ARM}_xdom-${D}_seed${SEED}"; mkdir -p "$OUT"; plan_receipt "$OUT"
"$PLAN_PY" "$M3CODE/m31_tagger.py" --tag "$TAG" --seed "$SEED" --out "$OUT" --exclude-resource "$D" $EXTRA
plan_done "$OUT"
