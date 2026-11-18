#!/bin/bash
# MARS-C B3 (step 2 of 2) -- the div1 emission rule, the evaluator, and the OBSERVED contrasts.
#
# Depends on mc_b3_score.sbatch (afterok). Reads only score files that job wrote plus the two frozen
# inventories, the two ACU truth files and the already-published deployed cells; trains nothing.
#
# Three things this job is careful about, each of which has already destroyed a result in this project:
#   1. Every emitter is an EXPLICIT COMMA MANIFEST built by roster(), never a glob, and every eval runs
#      --strict-seeds --expect-seeds 3. A `*_seedavg_div1.jsonl` swept into a `*_seed*_div1.jsonl` roster
#      invalidated an entire results generation.
#   2. `paired_recall.mean` in the evaluator's output is the BOOTSTRAP mean, not the observed contrast.
#      b3_contrast.py recomputes the observed difference from --dump-per-pair and prints both, with the gap.
#   3. The bootstrap clusters on common.doc_key(source), not pair_id; b3_contrast.py re-derives that clustering
#      independently from the dump's own `doc` field and cross-checks its point estimates against the
#      evaluator's JSON to 1e-9 before reporting anything.
#
# Eval set, per pool, in descending priority so that a wall-clock casualty costs the least important table:
#   eval_<pool>_A    arm A human twin (primary) vs arm A judge twin   <- the headline matched contrast
#   eval_<pool>_R    arm R human twin (primary) vs arm R judge twin   <- the sampler control, reported apart
#   eval_<pool>_ctl  arm A, both twins, each with its shuffled-summary and no-summary B0 control
#   eval_<pool>_ctx  arm A matched twins beside the deployed humanfact / judgefact cells, same scale
#SBATCH --job-name=mars-b3-eval
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=05:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[b3] REFUSED: MC_TEST_REREAD=yes not set"; exit 3; }

B3="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}/b3"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
mkdir -p "$B3/perpair" "$B3/analysis"; plan_receipt "$B3"

INV_T="$M3/m32/units_gemma+ent_test.jsonl";        ACU_T="$G/b18/acu_units_test.jsonl";           DEP_T="$MC/g1_test/val"
INV_U="$MC/g7_unisum/units_gemma+ent_unisum.jsonl"; ACU_U="$MC/g7_unisum/acu_units_unisum.jsonl";  DEP_U="$MC/g7_unisum/val"
for F in "$INV_T" "$ACU_T" "$INV_U" "$ACU_U"; do [ -f "$F" ] || { echo "[b3] MISSING $F" >&2; exit 4; }; done

TAGS="b3Ahuman b3Ajudge b3Rhuman b3Rjudge b3Ahumanshuf b3Ahumanempty b3Ajudgeshuf b3Ajudgeempty"
# roster: build an explicit, order-fixed comma manifest and refuse on the first absent seed file.
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || { echo "[b3] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
EV () { "$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --k 10 --n-boot 2000 --robust-tests --n-perm 10000 "$@"; }

echo "===== B3.0 the div1 emission rule, applied identically to all 24 score files per pool"
for POOL in test unisum; do
  case "$POOL" in test) INV="$INV_T" ;; unisum) INV="$INV_U" ;; esac
  for T in $TAGS; do for S in $SEEDS; do
    F="$B3/scores_${POOL}/gemma+ent_${T}_seed${S}.jsonl"
    [ -s "$F" ] || { echo "[b3] MISSING score file $F (did mc_b3_score task fail?)" >&2; exit 4; }
  done; done
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "gemma+ent=$INV" --cap 1 \
    --scores-glob "$B3/scores_${POOL}/gemma+ent_b3*_seed*.jsonl" --out "$B3/val_${POOL}"
  for T in $TAGS; do roster "$B3/val_${POOL}" "gemma+ent_${T}" "_div1.jsonl" >/dev/null; done
  echo "[b3] $POOL: 24 div1 files present"
done

run_pool () {   # $1 pool  $2 inventory  $3 acu  $4 deployed val dir
  local POOL="$1" INV="$2" ACU="$3" DEP="$4" V="$B3/val_$1"
  echo "===== B3.1 $POOL -- arm A, the matched label-source contrast (headline)"
  EV --acu "$ACU" --primary "b3.A.human+div1" --out "$B3/eval_${POOL}_A" \
     --dump-per-pair "$B3/perpair/${POOL}_A.jsonl" --systems \
     "b3.A.human+div1=$INV:$(roster "$V" "gemma+ent_b3Ahuman" "_div1.jsonl")" \
     "b3.A.judge+div1=$INV:$(roster "$V" "gemma+ent_b3Ajudge" "_div1.jsonl")"
  echo "===== B3.2 $POOL -- arm R, the random-natural-cell sampler control, reported separately"
  EV --acu "$ACU" --primary "b3.R.human+div1" --out "$B3/eval_${POOL}_R" \
     --dump-per-pair "$B3/perpair/${POOL}_R.jsonl" --systems \
     "b3.R.human+div1=$INV:$(roster "$V" "gemma+ent_b3Rhuman" "_div1.jsonl")" \
     "b3.R.judge+div1=$INV:$(roster "$V" "gemma+ent_b3Rjudge" "_div1.jsonl")"
}
run_ctl () {
  local POOL="$1" INV="$2" ACU="$3" DEP="$4" V="$B3/val_$1"
  echo "===== B3.3 $POOL -- the standing B0 summary-blind controls on both matched twins"
  EV --acu "$ACU" --primary "b3.A.human+div1" --out "$B3/eval_${POOL}_ctl" \
     --dump-per-pair "$B3/perpair/${POOL}_ctl.jsonl" --systems \
     "b3.A.human+div1=$INV:$(roster "$V" "gemma+ent_b3Ahuman" "_div1.jsonl")" \
     "b3.A.human.shuffled+div1=$INV:$(roster "$V" "gemma+ent_b3Ahumanshuf" "_div1.jsonl")" \
     "b3.A.human.empty+div1=$INV:$(roster "$V" "gemma+ent_b3Ahumanempty" "_div1.jsonl")" \
     "b3.A.judge+div1=$INV:$(roster "$V" "gemma+ent_b3Ajudge" "_div1.jsonl")" \
     "b3.A.judge.shuffled+div1=$INV:$(roster "$V" "gemma+ent_b3Ajudgeshuf" "_div1.jsonl")" \
     "b3.A.judge.empty+div1=$INV:$(roster "$V" "gemma+ent_b3Ajudgeempty" "_div1.jsonl")"
  echo "===== B3.4 $POOL -- the matched twins beside the deployed cells, on one scale"
  EV --acu "$ACU" --primary "b3.A.human+div1" --out "$B3/eval_${POOL}_ctx" \
     --dump-per-pair "$B3/perpair/${POOL}_ctx.jsonl" --systems \
     "b3.A.human+div1=$INV:$(roster "$V" "gemma+ent_b3Ahuman" "_div1.jsonl")" \
     "b3.A.judge+div1=$INV:$(roster "$V" "gemma+ent_b3Ajudge" "_div1.jsonl")" \
     "deployed.humanfact+div1=$INV:$(roster "$DEP" "gemma+ent_humanfact" "_div1.jsonl")" \
     "deployed.judgefact+div1=$INV:$(roster "$DEP" "gemma+ent_xenc" "_div1.jsonl")"
}
run_pool test   "$INV_T" "$ACU_T" "$DEP_T"
run_pool unisum "$INV_U" "$ACU_U" "$DEP_U"
run_ctl  test   "$INV_T" "$ACU_T" "$DEP_T"
run_ctl  unisum "$INV_U" "$ACU_U" "$DEP_U"

echo "===== B3.5 observed contrasts with document-clustered intervals (NOT the bootstrap mean)"
SPECS=()
for POOL in test unisum; do for W in A R ctl ctx; do
  SPECS+=("${POOL}_${W}=$B3/eval_${POOL}_${W}:$B3/perpair/${POOL}_${W}.jsonl")
done; done
"$PLAN_PY" "$SCODE/b3_contrast.py" --n-boot 2000 --boot-seed 20260918 --skip-missing \
  --eval "${SPECS[@]}" --out "$B3/analysis/b3_contrasts.json"

(cd "$B3" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$B3"
echo "[b3] done"
