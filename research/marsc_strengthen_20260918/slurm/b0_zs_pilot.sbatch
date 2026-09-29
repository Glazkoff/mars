#!/bin/bash
# MARS-C strengthen B0 -- throughput pilot for the off-the-shelf control harness.
#
# The cross-encoder families have a measured cost (0.00083 GPU-s/unit); the zero-shot families do not, and
# SummaC-ZS in particular runs one DeBERTa-large fp32 forward per (summary sentence x claim sentence) pair, so
# its per-unit cost is not a constant of the model but of the summary. This job scores 200 pairs of the
# validation distilled inventory under all three candidate modes and prints units/s per system, so the full
# grid can be sized from measurement rather than from a guess. Nothing here is analysed.
#
#   sbatch research/marsc_strengthen_20260918/slurm/b0_zs_pilot.sbatch
#SBATCH --job-name=mars-b0-zs-pilot
#SBATCH --partition=train
#SBATCH --nodelist=Euler
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
LIM="${B0_PILOT_LIMIT:-200}"
INVFILE="$M3/m32/units_distilled_validation.jsonl"; [ -f "$INVFILE" ] || { echo "[b0p] MISSING $INVFILE"; exit 4; }
echo "[b0p] pilot limit=$LIM units=$INVFILE device=$( [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && echo cuda || echo cpu )"
OUT="$MCS/b0/pilot"; mkdir -p "$OUT/scores"; plan_receipt "$OUT"
for MODE in real shuffled empty; do
  echo "===== pilot candidate-mode=$MODE"
  "$PLAN_PY" "$SCODE/b0_zs_score_units.py" --units "distilled=$INVFILE" \
    --systems lex,minilm,nli,summac_zs --candidate-mode "$MODE" --shuffle-seed 20260917 \
    --scores-out "$OUT/scores" --batch 32 --limit "$LIM"
done
echo "===== measured throughput: read the [b0zs] ... done in Ns (M units/s) lines above"
plan_done "$OUT"
echo "[b0p] done -> $OUT"
