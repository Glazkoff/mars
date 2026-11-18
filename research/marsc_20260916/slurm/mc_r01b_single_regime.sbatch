#!/bin/bash
# MARS-C R01b -- the single-model regime's own paired intervals, re-run under the hardened evaluator so that
# every interval in the paper carries a recorded emitter manifest. These cells were never affected by the
# consensus-glob defect (one seed-averaged file each); the numbers are expected to reproduce exactly.
# Also: the validation-split comparison and the label-efficiency curve, for the same reason.
#SBATCH --job-name=mc-r01b-single
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[r01b] REFUSED"; exit 3; }
G1="$MC/g1_test"; G7="$MC/g7_unisum"; DIV="$MC/e2div"; OUT="$MC/r01_consensus"; mkdir -p "$OUT"
INV_GE="$M3/m32/units_gemma+ent_test.jsonl"; INV_DI="$M3/m32/units_distilled_test.jsonl"; ACU_T="$G/b18/acu_units_test.jsonl"
INV_GE_V="$M3/m32/units_gemma+ent_validation.jsonl"; INV_DI_V="$M3/m32/units_distilled_validation.jsonl"; ACU_V="$G/b18/acu_units_validation.jsonl"
INV_U="$G7/units_gemma+ent_unisum.jsonl"; ACU_U="$G7/acu_units_unisum.jsonl"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || { echo "[r01b] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
EV () { "$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 "$@"; }

echo "===== R01b.1 sealed TEST, single-model regime as primary"
EV --acu "$ACU_T" --k 5 10 --n-boot 500 --robust-tests --n-perm 10000 --primary "humanfact.single+div1" \
   --out "$OUT/eval_test_single" --systems \
   "humanfact.single+div1=$INV_GE:$G1/val/gemma+ent_humanfact_seedavg_div1.jsonl" \
   "judgefact.distilled.single+div1=$INV_DI:$G1/val/distilled_xenc_seedavg_div1.jsonl" \
   "humanfact.distilled.single+div1=$INV_DI:$G1/val/distilled_humanfact_seedavg_div1.jsonl" \
   "judgefact.single+div1=$INV_GE:$G1/val/gemma+ent_xenc_seedavg_div1.jsonl" 2>&1 | grep -E "\[e2"

echo "===== R01b.2 UniSumEval, single-model regime as primary (fixed and adaptive budgets)"
EV --acu "$ACU_U" --k 5 10 --n-boot 500 --robust-tests --n-perm 10000 --primary "humanfact.single+div1" \
   --out "$OUT/eval_unisum_single" --systems \
   "humanfact.single+div1=$INV_U:$G7/val/gemma+ent_humanfact_seedavg_div1.jsonl" \
   "judgefact.single+div1=$INV_U:$G7/val/gemma+ent_xenc_seedavg_div1.jsonl" 2>&1 | grep -E "\[e2"
EV --acu "$ACU_U" --k 10 --adaptive-k "sent:0.50" --ak-min 3 --ak-max 50 --n-boot 500 --primary "humanfact.single+div1" \
   --out "$OUT/eval_unisum_single_adaptive" --systems \
   "humanfact.single+div1=$INV_U:$G7/val/gemma+ent_humanfact_seedavg_div1.jsonl" \
   "judgefact.single+div1=$INV_U:$G7/val/gemma+ent_xenc_seedavg_div1.jsonl" \
   "humanfact+div1=$INV_U:$(roster "$G7/val" "gemma+ent_humanfact" "_div1.jsonl")" \
   "judgefact+div1=$INV_U:$(roster "$G7/val" "gemma+ent_xenc" "_div1.jsonl")" 2>&1 | grep -E "\[e2"

echo "===== R01b.3 the validation split (where the pipelines were developed)"
EV --acu "$ACU_V" --k 5 10 --n-boot 500 --primary "humanfact+div1" --out "$OUT/eval_validation" --systems \
   "humanfact+div1=$INV_GE_V:$(roster "$DIV/val" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_div1.jsonl")" \
   "humanfact.distilled+div1=$INV_DI_V:$(roster "$DIV/val" "distilled_humanfact" "_div1.jsonl")" \
   "judgefact.distilled+div1=$INV_DI_V:$(roster "$DIV/val" "units_distilled_xenc" "_div1.jsonl")" 2>&1 | grep -E "\[e2"

echo "===== R01b.4 label-efficiency curve (T07), re-run under the hardened evaluator"
T7="$MC/t07_curve"
SPECS=""
for FR in 10 25 50 100; do
  [ -f "$T7/val/gemma+ent_t07-frac${FR}_seed20260906_div1.jsonl" ] || continue
  SPECS="$SPECS frac${FR}+div1=$INV_GE_V:$(roster "$T7/val" "gemma+ent_t07-frac${FR}" "_div1.jsonl")"
done
if [ -n "$SPECS" ]; then
  EV --acu "$ACU_V" --k 5 10 --n-boot 2000 --primary "frac100+div1" --dump-per-pair "$OUT/per_pair_curve.jsonl" \
     --out "$OUT/eval_curve" --systems $SPECS 2>&1 | grep -E "\[e2"
fi
(cd "$OUT" && find . -type f -name "*.json*" | sort | xargs sha256sum > SHA256SUMS)
echo "[r01b] done"
