#!/bin/bash
# MARS-C strengthening campaign, Wave A blocks A4 + A5 + A6 (research/marsc_strengthen_20260918/PREREG.md).
#
# A4 -- the matching rule is ASYMMETRIC. mc_e2_eval.py builds the ACU side of the token filter content-only
#   (common.tokens(text, content_only=True), with an all-token fallback) but the EMITTED-UNIT side with all tokens
#   including stopwords. The >=1/>=2/>=3 shared-token floors cannot see that (a content-only ACU set has no stopword
#   to intersect), but the Jaccard floor can: the emitted unit's stopwords enter the UNION only. That is why the
#   Jaccard-floor sensitivity rows were withdrawn from the paper. a4_eval_symmetric.py is a fork of the evaluator
#   whose ONLY behavioural difference is an opt-in --match-symmetric flag; this job re-runs the whole R11 matching
#   sensitivity under BOTH rules, on validation AND on the sealed TEST split AND on the external UniSumEval pool,
#   from explicit three-seed manifests under --strict-seeds. No bar: a repair reports the corrected number.
#   The >=1/>=2/>=3 rows are expected to be numerically unchanged; running them under both rules is the evidence.
#
# A5 -- measure the seedrank x div1 incompatibility from the actual score files (provenance item 4c). The -5.0
#   diversity push is calibrated for probability-scale scores; seedrank scores are negated mean ranks. Audited
#   against the probability-scale seedavg arm on the same pool and the same emission step.
#
# A6 -- recompute the B1 judge-label census ("85,600 cells over 2,625 pairs", provenance C06) directly from
#   training_rows.jsonl using b21_finetune_xenc.py's own selection, and report the recomputed number whatever it is.
#
# Sealed-TEST gate: this job re-reads the held-out split, so it refuses unless MC_TEST_REREAD=yes is exported.
# PREREG.md (2026-09-18) is the dated amendment that authorises it.
#SBATCH --job-name=mars-a4a6
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
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[a4a6] REFUSED"; exit 3; }
OUT="${MCS_OUT:-/home/user/results/marsc_strengthen}/a4a6"; mkdir -p "$OUT"; plan_receipt "$OUT"

ACU_V="$G/b18/acu_units_validation.jsonl"; ACU_T="$G/b18/acu_units_test.jsonl"
G7="$MC/g7_unisum"; ACU_U="$G7/acu_units_unisum.jsonl"
INV_GE_V="$M3/m32/units_gemma+ent_validation.jsonl"; INV_DI_V="$M3/m32/units_distilled_validation.jsonl"
INV_GE_T="$M3/m32/units_gemma+ent_test.jsonl";       INV_DI_T="$M3/m32/units_distilled_test.jsonl"
INV_U="$G7/units_gemma+ent_unisum.jsonl"
G1="$MC/g1_test"; DIV="$MC/e2div"; DF="$MC/e2div_full"; T08="$MC/t08_agg"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || { echo "[a4a6] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }

echo "===== A6 -- recompute the B1 judge-label census from the training rows"
"$PLAN_PY" "$SCODE/a6_b1_census.py" --out "$OUT" \
  --train "$B1_DIR/training_rows.jsonl" --claim-cells 85600 --claim-pairs 2625 \
  --configs "$G/b21/xenc_seed20260906/training_config.json" \
            "$G/b21/xenc_seed20260907/training_config.json" \
            "$G/b21/xenc_seed20260908/training_config.json"

echo "===== A5 -- seedrank x div1 incompatibility, against the probability-scale seedavg control"
"$PLAN_PY" "$SCODE/a5_seedrank_audit.py" --k 10 --out "$OUT" \
  --arm "seedrank.humanfact.gemma+ent.TEST=$INV_GE_T:$T08/inv/gemma+ent_humanfact_seedrank.jsonl:$T08/val/gemma+ent_humanfact_seedrank_div1.jsonl" \
  --arm "seedlead.humanfact.gemma+ent.TEST=$INV_GE_T:$T08/inv/gemma+ent_humanfact_seedlead.jsonl:$T08/val/gemma+ent_humanfact_seedlead_div1.jsonl" \
  --arm "seedavg.humanfact.gemma+ent.TEST=$INV_GE_T:$G1/inv/gemma+ent_humanfact_seedavg.jsonl:$G1/val/gemma+ent_humanfact_seedavg_div1.jsonl" \
  --arm "seedrank.xenc.distilled.TEST=$INV_DI_T:$T08/inv/distilled_xenc_seedrank.jsonl:$T08/val/distilled_xenc_seedrank_div1.jsonl" \
  --arm "seedavg.xenc.distilled.TEST=$INV_DI_T:$G1/inv/distilled_xenc_seedavg.jsonl:$G1/val/distilled_xenc_seedavg_div1.jsonl" \
  --arm "seedrank.humanfact.gemma+ent.UNISUM=$INV_U:$T08/inv/gemma+ent_humanfactU_seedrank.jsonl:$T08/val/gemma+ent_humanfactU_seedrank_div1.jsonl" \
  --arm "seedavg.humanfact.gemma+ent.UNISUM=$INV_U:$G7/inv/gemma+ent_humanfact_seedavg.jsonl:$G7/val/gemma+ent_humanfact_seedavg_div1.jsonl"

echo "===== A4 -- rosters (explicit three-seed manifests; --strict-seeds --expect-seeds 3)"
SPECS_V="humanfact+div1=$INV_GE_V:$(roster "$DIV/val" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_div1.jsonl") \
humanfact.distilled+div1=$INV_DI_V:$(roster "$DIV/val" "distilled_humanfact" "_div1.jsonl") \
judgefact.distilled+div1=$INV_DI_V:$(roster "$DIV/val" "units_distilled_xenc" "_div1.jsonl") \
judgefact+div1=$INV_GE_V:$(roster "$DF/val" "units_gemma+ent_xenc" "_div1.jsonl")"
SPECS_T="humanfact+div1=$INV_GE_T:$(roster "$G1/val" "gemma+ent_humanfact" "_div1.jsonl") \
judgefact.distilled+div1=$INV_DI_T:$(roster "$G1/val" "distilled_xenc" "_div1.jsonl") \
humanfact.distilled+div1=$INV_DI_T:$(roster "$G1/val" "distilled_humanfact" "_div1.jsonl") \
judgefact+div1=$INV_GE_T:$(roster "$G1/val" "gemma+ent_xenc" "_div1.jsonl") \
humanfact.single+div1=$INV_GE_T:$G1/val/gemma+ent_humanfact_seedavg_div1.jsonl \
judgefact.distilled.single+div1=$INV_DI_T:$G1/val/distilled_xenc_seedavg_div1.jsonl"
SPECS_U="humanfact+div1=$INV_U:$(roster "$G7/val" "gemma+ent_humanfact" "_div1.jsonl") \
judgefact+div1=$INV_U:$(roster "$G7/val" "gemma+ent_xenc" "_div1.jsonl") \
humanfact.single+div1=$INV_U:$G7/val/gemma+ent_humanfact_seedavg_div1.jsonl \
judgefact.single+div1=$INV_U:$G7/val/gemma+ent_xenc_seedavg_div1.jsonl"

run_eval () {   # $1 rule flag ("" | --match-symmetric)  $2 rule tag  $3 pool tag  $4 acu  $5 min-tokens  $6 jaccard  $7 cfg tag  $8.. specs
  local flag="$1" rule="$2" pool="$3" acu="$4" mt="$5" jc="$6" cfg="$7"; shift 7
  local log="$OUT/logs/${rule}_${pool}_${cfg}.log"; mkdir -p "$OUT/logs"
  # Full stdout (including the per-emitter [e2:manifest] provenance lines) is kept in $log; only the summary
  # rows are echoed to the job log. A python failure propagates: no `| grep` swallowing the exit status.
  "$PLAN_PY" "$SCODE/a4_eval_symmetric.py" --strict-seeds --expect-seeds 3 $flag \
    --match-min-tokens "$mt" --match-min-jaccard "$jc" \
    --acu "$acu" --k 10 --n-boot 500 --primary "humanfact+div1" \
    --out "$OUT/eval_${rule}_${pool}_${cfg}" --systems "$@" > "$log" 2>&1
  grep -E "^\[e2 k=" "$log" || echo "[a4a6] WARNING no [e2 k=] rows in $log"
}

for RULE in asym sym; do
  if [ "$RULE" = "sym" ]; then FLAG="--match-symmetric"; else FLAG=""; fi   # not `&&`: an AND-list returning 1 exits under set -e
  for CFG in "1 0.0 t1" "2 0.0 t2" "3 0.0 t3" "1 0.20 j20" "1 0.34 j34"; do
    set -- $CFG
    echo "===== A4 rule=$RULE match-min-tokens=$1 match-min-jaccard=$2"
    echo "----- validation"; run_eval "$FLAG" "$RULE" validation "$ACU_V" "$1" "$2" "$3" $SPECS_V
    echo "----- test";       run_eval "$FLAG" "$RULE" test       "$ACU_T" "$1" "$2" "$3" $SPECS_T
    echo "----- unisum";     run_eval "$FLAG" "$RULE" unisum     "$ACU_U" "$1" "$2" "$3" $SPECS_U
  done
done

(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[a4a6] done -> $OUT"
