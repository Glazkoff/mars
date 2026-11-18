#!/bin/bash
% SUPERSEDED-BY: mc_r01_consensus_fix.sbatch
% This job selected its score files with a wildcard that matched more files than it intended (the
% pseudo-seed *_seedavg_div1.jsonl, or the importance-policy outputs *_imp0.2_div1.jsonl). It is kept
% unchanged as the record of what was actually run; its numbers are superseded. Do not re-run it to
% reproduce the paper -- run the job named above, which reads explicit three-file manifests.
# MARS-C T05 -- MARS-2 as an emitter on the sealed TEST split, so the headline table carries our own predecessor.
# The TEST dumps are produced by slurm/cluster-B/plan2026/b2_score.sbatch with B2_SPLIT=test (weights exist in
# models/plan2026/b2); this job converts them to the unit_scores format, applies the identical emission rule and
# re-runs the evaluation. Registered as ADDING A BASELINE to a sealed split, not as re-reading our own result.
#SBATCH --job-name=mc-t05-mars2
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
G1="$MC/g1_test"; G7="$MC/g7_unisum"
INV_GE="$M3/m32/units_gemma+ent_test.jsonl"; INV_DI="$M3/m32/units_distilled_test.jsonl"; ACU_T="$G/b18/acu_units_test.jsonl"
INV_GE_V="$M3/m32/units_gemma+ent_validation.jsonl"; INV_DI_V="$M3/m32/units_distilled_validation.jsonl"; ACU_V="$G/b18/acu_units_validation.jsonl"
INV_U="$G7/units_gemma+ent_unisum.jsonl"; ACU_U="$G7/acu_units_unisum.jsonl"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[t] REFUSED: MC_TEST_REREAD=yes not set (TEST is sealed; this is a registered post-hoc SECONDARY endpoint on frozen outputs, user decision required)"; exit 3; }

OUT="$MC/t05_mars2"; mkdir -p "$OUT/inv" "$OUT/val"; plan_receipt "$OUT"
SRC_UNITS="${MC_T05_SRC_UNITS:-$ACU_T}"     # supplies source/candidate/resource context for the dump rows
EXTRA=""
for ARM in full_recovery direct_enabled; do
  DUMPS="$PLAN_OUT/b2/scores/${ARM}_seed*/units_test.jsonl"
  N=$(ls $DUMPS 2>/dev/null | wc -l); echo "[t05] $ARM: $N TEST dumps"
  [ "$N" -gt 0 ] || { echo "[t05] no TEST dumps for $ARM -- run b2_score.sbatch with B2_SPLIT=test first"; continue; }
  "$PLAN_PY" "$MCCODE/mc_variants.py" dumps2units --dumps "$DUMPS" --source-units "$SRC_UNITS" --name "mars2_$ARM" --split test --out "$OUT/inv"
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "mars2_$ARM=$OUT/inv/units_mars2_${ARM}_test.jsonl" --scores-glob "$OUT/inv/mars2_${ARM}_mars2_seed0.jsonl" --out "$OUT/val"
  EXTRA="$EXTRA mars2.${ARM}=$OUT/inv/units_mars2_${ARM}_test.jsonl:$OUT/inv/mars2_${ARM}_mars2_seed0.jsonl"
  EXTRA="$EXTRA mars2.${ARM}+div1=$OUT/inv/units_mars2_${ARM}_test.jsonl:$OUT/val/mars2_${ARM}_mars2_seed0_div1.jsonl"
done
[ -n "$EXTRA" ] || { echo "[t05] nothing to evaluate"; exit 4; }
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --acu "$ACU_T" --k 5 10 --n-boot 500 --primary "humanfact+div1" --out "$OUT/eval_test" --systems \
  "humanfact+div1=$INV_GE:$G1/val/gemma+ent_humanfact_seed*_div1.jsonl" "judgefact.distilled+div1=$INV_DI:$G1/val/distilled_xenc_seed*_div1.jsonl" \
  $EXTRA 2>&1 | grep "\[e2"
sha256sum "$OUT"/eval_test/*.json > "$OUT/SHA256SUMS"; plan_done "$OUT"
