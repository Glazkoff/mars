#!/bin/bash
# MARS-C R01 -- REPAIR of the consensus estimator (FULL_REVIEW_2026-09-17 decision-critical failure 1).
#
# Defect: G1/G7/T01-T05/T08 fed `*_seed*_div1.jsonl` to the evaluator. That glob matches the three per-seed
# diversity outputs AND `*_seedavg_div1.jsonl` -- the rule applied to the SEED-AVERAGED scores -- so every
# advertised "three-seed consensus" averaged four files, one of them already a function of the other three.
# Repair: every consensus emitter is now an EXPLICIT three-file manifest, and the evaluator runs with
# --strict-seeds --expect-seeds 3, which refuses a pseudo-seed inside a seed roster and refuses a roster that
# is not exactly three distinct numeric seeds. Nothing is re-scored: the frozen per-seed score files are the
# same bytes as in G1/G7 (their sha256 is recorded in the emitter manifest of every output json).
#
# Single-model cells (`*.single+div1`, one *_seedavg_div1.jsonl file) were already correct and are rerun
# unchanged so that both regimes come out of one evaluator build.
#SBATCH --job-name=mc-r01-consensus
#SBATCH --partition=infer
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
G1="$MC/g1_test"; G7="$MC/g7_unisum"
INV_GE="$M3/m32/units_gemma+ent_test.jsonl"; INV_DI="$M3/m32/units_distilled_test.jsonl"; ACU_T="$G/b18/acu_units_test.jsonl"
INV_U="$G7/units_gemma+ent_unisum.jsonl"; ACU_U="$G7/acu_units_unisum.jsonl"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[r01] REFUSED: MC_TEST_REREAD=yes not set (TEST is sealed; this is a REPAIR re-read of frozen outputs, user decision required)"; exit 3; }
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
OUT="$MC/r01_consensus"; mkdir -p "$OUT"; plan_receipt "$OUT"

# roster <dir> <prefix> <suffix>  ->  comma-separated manifest of exactly the registered seeds, asserted to exist
roster () {
  local dir="$1" pre="$2" suf="$3" out="" s f
  for s in $SEEDS; do
    f="$dir/${pre}_seed${s}${suf}"
    [ -f "$f" ] || { echo "[r01] MISSING seed file $f" >&2; exit 4; }
    out="${out:+$out,}$f"
  done
  printf '%s' "$out"
}
EV () { "$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 "$@"; }

# ---------------------------------------------------------------- RoSE sealed TEST (was G1)
HF_GE_C=$(roster "$G1/val" "gemma+ent_humanfact" "_div1.jsonl")
HF_DI_C=$(roster "$G1/val" "distilled_humanfact" "_div1.jsonl")
JF_GE_C=$(roster "$G1/val" "gemma+ent_xenc"      "_div1.jsonl")
JF_DI_C=$(roster "$G1/val" "distilled_xenc"      "_div1.jsonl")
HF_GE_U=$(roster "$G1/scores" "gemma+ent_humanfact" ".jsonl")
HF_DI_U=$(roster "$G1/scores" "distilled_humanfact" ".jsonl")
JF_GE_U=$(roster "$G1/scores" "gemma+ent_xenc"      ".jsonl")
JF_DI_U=$(roster "$G1/scores" "distilled_xenc"      ".jsonl")
SINGLE_T="humanfact.single+div1=$INV_GE:$G1/val/gemma+ent_humanfact_seedavg_div1.jsonl \
humanfact.distilled.single+div1=$INV_DI:$G1/val/distilled_humanfact_seedavg_div1.jsonl \
judgefact.single+div1=$INV_GE:$G1/val/gemma+ent_xenc_seedavg_div1.jsonl \
judgefact.distilled.single+div1=$INV_DI:$G1/val/distilled_xenc_seedavg_div1.jsonl"
GRID_T="humanfact+div1=$INV_GE:$HF_GE_C humanfact.distilled+div1=$INV_DI:$HF_DI_C \
judgefact+div1=$INV_GE:$JF_GE_C judgefact.distilled+div1=$INV_DI:$JF_DI_C \
humanfact=$INV_GE:$HF_GE_U humanfact.distilled=$INV_DI:$HF_DI_U \
judgefact=$INV_GE:$JF_GE_U judgefact.distilled=$INV_DI:$JF_DI_U \
humanfact.single=$INV_GE:$G1/inv/gemma+ent_humanfact_seedavg.jsonl \
humanfact.distilled.single=$INV_DI:$G1/inv/distilled_humanfact_seedavg.jsonl \
judgefact.single=$INV_GE:$G1/inv/gemma+ent_xenc_seedavg.jsonl \
judgefact.distilled.single=$INV_DI:$G1/inv/distilled_xenc_seedavg.jsonl $SINGLE_T"

echo "===== R01.1 sealed TEST, headline k=5/10 + emitted dump (replaces G1 eval_test)"
EV --acu "$ACU_T" --k 5 10 20 --n-boot 500 --primary "humanfact+div1" --robust-tests --n-perm 10000 \
   --dump-emitted "$OUT/emitted_test.jsonl" --dump-per-pair "$OUT/per_pair_test.jsonl" --out "$OUT/eval_test" --systems \
   $GRID_T 2>&1 | grep -E "\[e2"

echo "===== R01.2 sealed TEST, k curve (replaces T02 test half)"
EV --acu "$ACU_T" --k 1 3 5 10 20 50 --n-boot 500 --primary "humanfact+div1" --out "$OUT/eval_test_kcurve" --systems \
   "humanfact+div1=$INV_GE:$HF_GE_C" "humanfact.distilled+div1=$INV_DI:$HF_DI_C" \
   "judgefact.distilled+div1=$INV_DI:$JF_DI_C" "judgefact+div1=$INV_GE:$JF_GE_C" \
   "humanfact.single+div1=$INV_GE:$G1/val/gemma+ent_humanfact_seedavg_div1.jsonl" \
   "judgefact.distilled.single+div1=$INV_DI:$G1/val/distilled_xenc_seedavg_div1.jsonl" \
   "humanfact=$INV_GE:$HF_GE_U" "judgefact.distilled=$INV_DI:$JF_DI_U" 2>&1 | grep -E "\[e2"

echo "===== R01.3 sealed TEST, adaptive budget (replaces T03 test half)"
EV --acu "$ACU_T" --k 10 --adaptive-k "sent:0.25,sent:0.50" --ak-min 3 --ak-max 50 --n-boot 500 \
   --primary "humanfact+div1" --out "$OUT/eval_test_adaptive" --systems \
   "humanfact+div1=$INV_GE:$HF_GE_C" "humanfact.distilled+div1=$INV_DI:$HF_DI_C" \
   "judgefact.distilled+div1=$INV_DI:$JF_DI_C" 2>&1 | grep -E "\[e2"

# ---------------------------------------------------------------- UniSumEval (was G7)
HF_U_C=$(roster "$G7/val" "gemma+ent_humanfact" "_div1.jsonl")
JF_U_C=$(roster "$G7/val" "gemma+ent_xenc"      "_div1.jsonl")
HF_U_U=$(roster "$G7/scores" "gemma+ent_humanfact" ".jsonl")
JF_U_U=$(roster "$G7/scores" "gemma+ent_xenc"      ".jsonl")
SINGLE_U="humanfact.single+div1=$INV_U:$G7/val/gemma+ent_humanfact_seedavg_div1.jsonl \
judgefact.single+div1=$INV_U:$G7/val/gemma+ent_xenc_seedavg_div1.jsonl"

echo "===== R01.4 UniSumEval, headline k=5/10 (replaces G7 eval_unisum)"
EV --acu "$ACU_U" --k 5 10 20 --n-boot 500 --primary "humanfact+div1" --robust-tests --n-perm 10000 \
   --dump-emitted "$OUT/emitted_unisum.jsonl" --dump-per-pair "$OUT/per_pair_unisum.jsonl" --out "$OUT/eval_unisum" --systems \
   "humanfact+div1=$INV_U:$HF_U_C" "judgefact+div1=$INV_U:$JF_U_C" \
   "humanfact=$INV_U:$HF_U_U" "judgefact=$INV_U:$JF_U_U" $SINGLE_U 2>&1 | grep -E "\[e2"

echo "===== R01.5 UniSumEval, k curve + adaptive budget (replaces T02/T03 unisum halves)"
EV --acu "$ACU_U" --k 1 3 5 10 20 50 --n-boot 500 --primary "humanfact+div1" --out "$OUT/eval_unisum_kcurve" --systems \
   "humanfact+div1=$INV_U:$HF_U_C" "judgefact+div1=$INV_U:$JF_U_C" $SINGLE_U \
   "humanfact=$INV_U:$HF_U_U" "judgefact=$INV_U:$JF_U_U" 2>&1 | grep -E "\[e2"
EV --acu "$ACU_U" --k 10 --adaptive-k "sent:0.25,sent:0.50" --ak-min 3 --ak-max 50 --n-boot 500 \
   --primary "humanfact+div1" --dump-per-pair "$OUT/per_pair_unisum_adaptive.jsonl" --out "$OUT/eval_unisum_adaptive" --systems \
   "humanfact+div1=$INV_U:$HF_U_C" "judgefact+div1=$INV_U:$JF_U_C" $SINGLE_U \
   "humanfact=$INV_U:$HF_U_U" 2>&1 | grep -E "\[e2"

# ---------------------------------------------------------------- MARS-2 on TEST (was T05)
T5="$MC/t05_mars2"
EXTRA=""
for ARM in full_recovery direct_enabled; do
  [ -f "$T5/inv/units_mars2_${ARM}_test.jsonl" ] || continue
  EXTRA="$EXTRA mars2.${ARM}=$T5/inv/units_mars2_${ARM}_test.jsonl:$T5/inv/mars2_${ARM}_mars2_seed0.jsonl"
  EXTRA="$EXTRA mars2.${ARM}+div1=$T5/inv/units_mars2_${ARM}_test.jsonl:$T5/val/mars2_${ARM}_mars2_seed0_div1.jsonl"
done
if [ -n "$EXTRA" ]; then
  echo "===== R01.6 MARS-2 predecessor on TEST (replaces T05)"
  EV --acu "$ACU_T" --k 5 10 --n-boot 500 --primary "humanfact+div1" --out "$OUT/eval_test_mars2" --systems \
     "humanfact+div1=$INV_GE:$HF_GE_C" "humanfact.distilled+div1=$INV_DI:$HF_DI_C" \
     "judgefact.distilled+div1=$INV_DI:$JF_DI_C" \
     "humanfact.single+div1=$INV_GE:$G1/val/gemma+ent_humanfact_seedavg_div1.jsonl" \
     "humanfact.distilled.single+div1=$INV_DI:$G1/val/distilled_humanfact_seedavg_div1.jsonl" \
     $EXTRA 2>&1 | grep -E "\[e2"
fi

# ---------------------------------------------------------------- seed aggregation alternatives (was T08)
T8="$MC/t08_agg"
if [ -f "$T8/val/gemma+ent_humanfact_seedrank_div1.jsonl" ]; then
  echo "===== R01.7 seed-aggregation alternatives (replaces T08)"
  EV --acu "$ACU_T" --k 5 10 --n-boot 500 --primary "humanfact+div1" --out "$OUT/eval_test_agg" --systems \
     "humanfact+div1=$INV_GE:$HF_GE_C" \
     "humanfact.single+div1=$INV_GE:$G1/val/gemma+ent_humanfact_seedavg_div1.jsonl" \
     "humanfact.rank+div1=$INV_GE:$T8/val/gemma+ent_humanfact_seedrank_div1.jsonl" \
     "humanfact.lead+div1=$INV_GE:$T8/val/gemma+ent_humanfact_seedlead_div1.jsonl" \
     "judgefact.distilled+div1=$INV_DI:$JF_DI_C" \
     "judgefact.distilled.rank+div1=$INV_DI:$T8/val/distilled_xenc_seedrank_div1.jsonl" \
     "judgefact.distilled.lead+div1=$INV_DI:$T8/val/distilled_xenc_seedlead_div1.jsonl" 2>&1 | grep -E "\[e2"
  EV --acu "$ACU_U" --k 5 10 --n-boot 500 --primary "humanfact+div1" --out "$OUT/eval_unisum_agg" --systems \
     "humanfact+div1=$INV_U:$HF_U_C" \
     "humanfact.single+div1=$INV_U:$G7/val/gemma+ent_humanfact_seedavg_div1.jsonl" \
     "humanfact.rank+div1=$INV_U:$T8/val/gemma+ent_humanfactU_seedrank_div1.jsonl" \
     "humanfact.lead+div1=$INV_U:$T8/val/gemma+ent_humanfactU_seedlead_div1.jsonl" \
     "judgefact+div1=$INV_U:$JF_U_C" \
     "judgefact.rank+div1=$INV_U:$T8/val/gemma+ent_xencU_seedrank_div1.jsonl" \
     "judgefact.lead+div1=$INV_U:$T8/val/gemma+ent_xencU_seedlead_div1.jsonl" 2>&1 | grep -E "\[e2"
fi

(cd "$OUT" && find . -type f -name "*.json*" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[r01] done -> $OUT"
