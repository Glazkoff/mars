#!/bin/bash
% SUPERSEDED-BY: mc_r01_consensus_fix.sbatch
% This job selected its score files with a wildcard that matched more files than it intended (the
% pseudo-seed *_seedavg_div1.jsonl, or the importance-policy outputs *_imp0.2_div1.jsonl). It is kept
% unchanged as the record of what was actually run; its numbers are superseded. Do not re-run it to
% reproduce the paper -- run the job named above, which reads explicit three-file manifests.
# MARS-C T04 -- distribution-free robustness for the headline contrast: document-level paired permutation test
# (10,000 sign-flips) and an exact per-document sign test, next to the registered bootstrap CI, on TEST and on
# UniSumEval. A +0.051 margin deserves more than one inferential procedure. CPU only.
#SBATCH --job-name=mc-t04-robust
#SBATCH --partition=infer
#SBATCH --gres=none
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
INV_GE_V="$M3/m32/units_gemma+ent_validation.jsonl"; INV_DI_V="$M3/m32/units_distilled_validation.jsonl"; ACU_V="$G/b18/acu_units_validation.jsonl"
INV_U="$G7/units_gemma+ent_unisum.jsonl"; ACU_U="$G7/acu_units_unisum.jsonl"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[t] REFUSED: MC_TEST_REREAD=yes not set (TEST is sealed; this is a registered post-hoc SECONDARY endpoint on frozen outputs, user decision required)"; exit 3; }

OUT="$MC/t04_robust"; mkdir -p "$OUT"; plan_receipt "$OUT"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --acu "$ACU_T" --k 5 10 --n-boot 500 --robust-tests --n-perm 10000 \
  --primary "humanfact+div1" --dump-per-pair "$OUT/per_pair_test.jsonl" --out "$OUT/eval_test" --systems \
  "humanfact+div1=$INV_GE:$G1/val/gemma+ent_humanfact_seed*_div1.jsonl" "judgefact.distilled+div1=$INV_DI:$G1/val/distilled_xenc_seed*_div1.jsonl" \
  "humanfact.distilled+div1=$INV_DI:$G1/val/distilled_humanfact_seed*_div1.jsonl" "judgefact+div1=$INV_GE:$G1/val/gemma+ent_xenc_seed*_div1.jsonl" 2>&1 | grep "\[e2"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --acu "$ACU_T" --k 10 --n-boot 500 --robust-tests --n-perm 10000 \
  --primary "humanfact.single+div1" --out "$OUT/eval_test_single" --systems \
  "humanfact.single+div1=$INV_GE:$G1/val/gemma+ent_humanfact_seedavg_div1.jsonl" "judgefact.distilled.single+div1=$INV_DI:$G1/val/distilled_xenc_seedavg_div1.jsonl" 2>&1 | grep "\[e2"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --acu "$ACU_U" --k 5 10 --n-boot 500 --robust-tests --n-perm 10000 \
  --primary "humanfact+div1" --dump-per-pair "$OUT/per_pair_unisum.jsonl" --out "$OUT/eval_unisum" --systems \
  "humanfact+div1=$INV_U:$G7/val/gemma+ent_humanfact_seed*_div1.jsonl" "judgefact+div1=$INV_U:$G7/val/gemma+ent_xenc_seed*_div1.jsonl" \
  "humanfact.single+div1=$INV_U:$G7/val/gemma+ent_humanfact_seedavg_div1.jsonl" "judgefact.single+div1=$INV_U:$G7/val/gemma+ent_xenc_seedavg_div1.jsonl" 2>&1 | grep "\[e2"
sha256sum "$OUT"/eval_*/*.json > "$OUT/SHA256SUMS"; plan_done "$OUT"
