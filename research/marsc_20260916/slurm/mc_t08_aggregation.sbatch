#!/bin/bash
% SUPERSEDED-BY: mc_r01_consensus_fix.sbatch
% This job selected its score files with a wildcard that matched more files than it intended (the
% pseudo-seed *_seedavg_div1.jsonl, or the importance-policy outputs *_imp0.2_div1.jsonl). It is kept
% unchanged as the record of what was actually run; its numbers are superseded. Do not re-run it to
% reproduce the paper -- run the job named above, which reads explicit three-file manifests.
# MARS-C T08 -- seed-aggregation variants: rank-average and lead-union, against the registered consensus and
# single-model regimes, on TEST and UniSumEval. G7 showed the single-model regime carries the LARGER external
# margin at a third of the cost, so the aggregation step is a free parameter and is measured, not assumed. CPU only.
#SBATCH --job-name=mc-t08-agg
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

OUT="$MC/t08_agg"; mkdir -p "$OUT/inv" "$OUT/val"; plan_receipt "$OUT"
agg () {  # $1 units-spec  $2 per-seed glob  $3 tag
  "$PLAN_PY" "$MCCODE/mc_aggregate.py" rankavg   --units "$1" --scores "$2" --tag "$3" --out "$OUT/inv"
  "$PLAN_PY" "$MCCODE/mc_aggregate.py" leadunion --units "$1" --scores "$2" --tag "$3" --out "$OUT/inv"
}
div () { "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "$1" --scores-glob "$2" --out "$OUT/val"; }
agg "gemma+ent=$INV_GE" "$G1/scores/gemma+ent_humanfact_seed*.jsonl" humanfact
agg "distilled=$INV_DI" "$G1/scores/distilled_xenc_seed*.jsonl" xenc
div "gemma+ent=$INV_GE" "$OUT/inv/gemma+ent_humanfact_seedrank.jsonl"; div "gemma+ent=$INV_GE" "$OUT/inv/gemma+ent_humanfact_seedlead.jsonl"
div "distilled=$INV_DI" "$OUT/inv/distilled_xenc_seedrank.jsonl"; div "distilled=$INV_DI" "$OUT/inv/distilled_xenc_seedlead.jsonl"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --acu "$ACU_T" --k 5 10 --n-boot 500 --primary "humanfact+div1" --out "$OUT/eval_test" --systems \
  "humanfact+div1=$INV_GE:$G1/val/gemma+ent_humanfact_seed*_div1.jsonl" \
  "humanfact.single+div1=$INV_GE:$G1/val/gemma+ent_humanfact_seedavg_div1.jsonl" \
  "humanfact.rank+div1=$INV_GE:$OUT/val/gemma+ent_humanfact_seedrank_div1.jsonl" \
  "humanfact.lead+div1=$INV_GE:$OUT/val/gemma+ent_humanfact_seedlead_div1.jsonl" \
  "judgefact.distilled+div1=$INV_DI:$G1/val/distilled_xenc_seed*_div1.jsonl" \
  "judgefact.distilled.rank+div1=$INV_DI:$OUT/val/distilled_xenc_seedrank_div1.jsonl" \
  "judgefact.distilled.lead+div1=$INV_DI:$OUT/val/distilled_xenc_seedlead_div1.jsonl" 2>&1 | grep "\[e2"
agg "gemma+ent=$INV_U" "$G7/scores/gemma+ent_humanfact_seed*.jsonl" humanfactU
agg "gemma+ent=$INV_U" "$G7/scores/gemma+ent_xenc_seed*.jsonl" xencU
for M in seedrank seedlead; do div "gemma+ent=$INV_U" "$OUT/inv/gemma+ent_humanfactU_${M}.jsonl"; div "gemma+ent=$INV_U" "$OUT/inv/gemma+ent_xencU_${M}.jsonl"; done
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --acu "$ACU_U" --k 5 10 --n-boot 500 --primary "humanfact+div1" --out "$OUT/eval_unisum" --systems \
  "humanfact+div1=$INV_U:$G7/val/gemma+ent_humanfact_seed*_div1.jsonl" \
  "humanfact.single+div1=$INV_U:$G7/val/gemma+ent_humanfact_seedavg_div1.jsonl" \
  "humanfact.rank+div1=$INV_U:$OUT/val/gemma+ent_humanfactU_seedrank_div1.jsonl" \
  "humanfact.lead+div1=$INV_U:$OUT/val/gemma+ent_humanfactU_seedlead_div1.jsonl" \
  "judgefact+div1=$INV_U:$G7/val/gemma+ent_xenc_seed*_div1.jsonl" \
  "judgefact.rank+div1=$INV_U:$OUT/val/gemma+ent_xencU_seedrank_div1.jsonl" \
  "judgefact.lead+div1=$INV_U:$OUT/val/gemma+ent_xencU_seedlead_div1.jsonl" 2>&1 | grep "\[e2"
sha256sum "$OUT"/eval_*/*.json > "$OUT/SHA256SUMS"; plan_done "$OUT"
