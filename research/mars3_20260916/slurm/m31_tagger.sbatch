#!/bin/bash
# M31 main arms: tasks 0-2 tagger, 3-5 tagger_pre, 6-8 tagger_srconly (seed = task % 3). Env: M31_EXTRA_UNITS="name=path ..." optional.
#SBATCH --job-name=m3-tagger
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-8
#SBATCH --output=/home/user/results/mars3/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
ARMS=(tagger tagger_pre tagger_srconly); read -r -a SEEDS <<< "${M31_SEEDS:-20260906 20260907 20260908}"; NS=${#SEEDS[@]}
T="${SLURM_ARRAY_TASK_ID:-0}"; ARM="${ARMS[$((T / NS))]}"; SEED="${SEEDS[$((T % NS))]}"
EXTRA=""; case "$ARM" in tagger_pre) EXTRA="--pretrain-rows $M3/m31/pseudo_all.jsonl";; tagger_srconly) EXTRA="--source-only";; esac
UNITS=""; for F in "$M3"/m32/units_*_validation.jsonl; do [ -s "$F" ] && UNITS="$UNITS $(basename "$F" _validation.jsonl)=$F"; done
OUT="$M3/m31/${ARM}_seed${SEED}"; mkdir -p "$OUT"; plan_receipt "$OUT"
"$PLAN_PY" "$M3CODE/m31_tagger.py" --tag "$ARM" --seed "$SEED" --out "$OUT" $EXTRA ${UNITS:+--eval-units $UNITS} $([ "$ARM" = tagger ] && echo --save-model)
plan_done "$OUT"
