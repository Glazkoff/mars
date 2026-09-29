#!/bin/bash
# M31 fact-level arms on the M32b rows: tasks 0-2 tagger_facts, 3-5 writer_facts (seed = task % 3).
#SBATCH --job-name=m3-facts
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/user/results/mars3/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
read -r -a SEEDS <<< "${M31_SEEDS:-20260906 20260907 20260908}"; NS=${#SEEDS[@]}
T="${SLURM_ARRAY_TASK_ID:-0}"; SEED="${SEEDS[$((T % NS))]}"; ROWS="$M3/m32/train_rows_facts.jsonl"
[ -s "$ROWS" ] || { echo "no M32b rows"; exit 3; }
UNITS=""; for F in "$M3"/m32/units_*_validation.jsonl; do [ -s "$F" ] && UNITS="$UNITS $(basename "$F" _validation.jsonl)=$F"; done
if [ $((T / NS)) = 0 ]; then
  OUT="$M3/m31/tagger_facts_seed${SEED}"; mkdir -p "$OUT"; plan_receipt "$OUT"
  "$PLAN_PY" "$M3CODE/m31_tagger.py" --tag tagger_facts --seed "$SEED" --out "$OUT" --train "$ROWS" ${UNITS:+--eval-units $UNITS} --save-model
else
  OUT="$M3/m31/writer_facts_seed${SEED}"; mkdir -p "$OUT"; plan_receipt "$OUT"
  "$PLAN_PY" "$M3CODE/m31_writer.py" --tag writer_facts --seed "$SEED" --out "$OUT" --train "$ROWS" --max-tgt 512 --beams 2
fi
plan_done "$OUT"
