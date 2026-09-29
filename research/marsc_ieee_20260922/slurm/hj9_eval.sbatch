#!/bin/bash
# H-J9 -- the frozen evaluator on both pools: the new FactCG-encoder arms (human and judge packages, with
# their summary-blind controls) beside the deployed MARS-C arms, the matched judge twin at the old backbone,
# and zero-shot FactCG / MiniCheck-7B / judge emissions already produced by B9/B10 (RoSE test) and H-J3b
# (UniSumEval). Then the OBSERVED document-clustered contrasts. Registered in PREREG.md amendment 6.
#
#   sbatch --dependency=afterok:<human>:<judge> --export=ALL,MC_TEST_REREAD=yes research/marsc_ieee_20260922/slurm/hj9_eval.sbatch
#SBATCH --job-name=mars-hj9-eval
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source "$PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh"
source "$PLAN_REPO/research/mars3_20260916/slurm/env.sh"
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[hj9e] REFUSED: needs MC_TEST_REREAD=yes"; exit 3; }
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
HJ9="${HJ9_ROOT:-$MCS/hj9}"; BB="${HJ9_BACKBONE:-factcg}"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; \
  [ -f "$f" ] || { echo "[hj9e] MISSING $f" >&2; return 1; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
EV () { "$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --k 10 --n-boot 2000 --robust-tests --n-perm 10000 "$@"; }

run_pool () {   # $1 pool
  local POOL="$1" INV ACU DEP JT CMP OUT="$HJ9/eval_$1"; mkdir -p "$OUT"; plan_receipt "$OUT"
  case "$POOL" in
    test)   INV="$M3/m32/units_gemma+ent_test.jsonl"; ACU="$G/b18/acu_units_test.jsonl"; DEP="$MC/g1_test/val"; JT="$MCS/b3/val_test"; CMP="$MCS/b9b10_eval/test/div" ;;
    unisum) INV="$MC/g7_unisum/units_gemma+ent_unisum.jsonl"; ACU="$MC/g7_unisum/acu_units_unisum.jsonl"; DEP="$MC/g7_unisum/val"; JT="$MCS/b3/val_unisum"; CMP="$MCS/hj3/eval_unisum/div" ;;
  esac
  for f in "$INV" "$ACU"; do [ -f "$f" ] || { echo "[hj9e] MISSING $f"; exit 4; }; done
  local SPECS=""
  for LAB in human judge; do
    local ST="hj9${BB}${LAB}"
    [ -f "$HJ9/val_${POOL}/gemma+ent_${ST}_seed20260906_div1.jsonl" ] || { echo "[hj9e] no $ST arms on $POOL (skipped)"; continue; }
    SPECS="$SPECS ${ST}+div1=$INV:$(roster "$HJ9/val_${POOL}" "gemma+ent_${ST}" "_div1.jsonl")"
    SPECS="$SPECS ${ST}.shuffled+div1=$INV:$(roster "$HJ9/val_${POOL}" "gemma+ent_${ST}shuf" "_div1.jsonl")"
    SPECS="$SPECS ${ST}.empty+div1=$INV:$(roster "$HJ9/val_${POOL}" "gemma+ent_${ST}empty" "_div1.jsonl")"
  done
  SPECS="$SPECS humanfact+div1=$INV:$(roster "$DEP" "gemma+ent_humanfact" "_div1.jsonl")"
  SPECS="$SPECS judgefact+div1=$INV:$(roster "$DEP" "gemma+ent_xenc" "_div1.jsonl")"
  SPECS="$SPECS matched.judge+div1=$INV:$(roster "$JT" "gemma+ent_b3Ajudge" "_div1.jsonl")"
  for f in "$CMP"/gemma+ent_nli_factcg_real_div1.jsonl "$CMP"/gemma+ent_nli_minicheck7b_real_div1.jsonl "$CMP"/gemma+ent_nli_judge_gemma31_real_div1.jsonl; do
    [ -e "$f" ] || { echo "[hj9e] MISSING comparator emission $f"; exit 4; }
    b=$(basename "$f" _div1.jsonl); SPECS="$SPECS ${b#gemma+ent_}+div1=$INV:$f"
  done
  echo "===== [hj9e] $POOL emitter set:"; for s in $SPECS; do echo "  ${s%%=*}"; done
  local PRIM="hj9${BB}human+div1"
  EV --acu "$ACU" --primary "$PRIM" --out "$OUT/eval_${POOL}" \
     --dump-per-pair "$OUT/per_pair_${POOL}_k10.jsonl" --dump-emitted "$OUT/emitted_${POOL}_k10.jsonl" \
     --systems $SPECS 2> >(tee -a "$OUT/e2_stderr.log" >&2) | grep -E "\[e2"
  "$PLAN_PY" "$SCODE/b3_contrast.py" --n-boot 2000 --boot-seed 20260918 --skip-missing \
    --eval "${POOL}_hj9=$OUT/eval_${POOL}:$OUT/per_pair_${POOL}_k10.jsonl" --out "$OUT/b3_contrasts.json"
  (cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
}
run_pool test
run_pool unisum
echo "[hj9e] done"
