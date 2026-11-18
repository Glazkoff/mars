#!/bin/bash
# H-J11 (gap fill, 2026-09-27) -- per-unit throughput of the seven off-the-shelf verifiers on DISTILLED units.
# B6 timed them on Gemma+ent only. Same harness (b6_cost.py time_offshelf), same 200-pair validation window, same
# unit caps; Gemma+ent is re-timed in the same run as a consistency check against the published column.
#SBATCH --job-name=mars-hj11-cost
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%j.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
OUT="$MCS/hj11/cost"; mkdir -p "$OUT"; plan_receipt "$OUT"
"$PLAN_PY" "$SCODE/b6_cost.py" --stage timings --which gpu --offshelf-only \
  --offshelf-inventories gemma+ent,distilled \
  --inv-gemma "$M3/m32/units_gemma+ent_validation.jsonl" --inv-distilled "$M3/m32/units_distilled_validation.jsonl" \
  --n-pairs 200 --offshelf lex,minilm,nli,summac_zs,minicheck,alignscore \
  --offshelf-unit-cap 6000 --slow-unit-cap 900 --skip-gemma \
  --timings-name ${HJ11_COST_NAME:-stage_times_offshelf_both.json} --out "$OUT"
plan_done "$OUT"; echo "[hj11] done -> $OUT"
