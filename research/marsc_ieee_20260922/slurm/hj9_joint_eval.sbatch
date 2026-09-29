#!/bin/bash
# H-J9 / H-J9b joint re-evaluation (no new scoring): every backbone arm of amendments 6 and 9 beside the
# pipeline of record, the matched judge twin and the three comparators, in ONE evaluator call per pool with
# the pipeline of record as primary, so every contrast in Table tab:marsc_backbone is "pipeline of record
# minus row" from a single bootstrap. Reads the div1 files the hj9 and hj9_xlarge chains already wrote.
# Post-hoc secondary; recorded under PREREG amendment 14.
#
#   sbatch --export=ALL,MC_TEST_REREAD=yes research/marsc_ieee_20260922/slurm/hj9_joint_eval.sbatch
#SBATCH --job-name=mars-hj9-joint
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source "$PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh"
source "$PLAN_REPO/research/mars3_20260916/slurm/env.sh"
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[hj9j] REFUSED: needs MC_TEST_REREAD=yes"; exit 3; }
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
J="$MCS/hj9_joint"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; \
  [ -f "$f" ] || { echo "[hj9j] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
EV () { "$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --k 10 --n-boot 2000 --robust-tests --n-perm 10000 "$@"; }

run_pool () {
  local POOL="$1" INV ACU DEP JT CMP OUT="$J/eval_$1"; mkdir -p "$OUT"; plan_receipt "$OUT"
  case "$POOL" in
    test)   INV="$M3/m32/units_gemma+ent_test.jsonl"; ACU="$G/b18/acu_units_test.jsonl"; DEP="$MC/g1_test/val"; JT="$MCS/b3/val_test"; CMP="$MCS/b9b10_eval/test/div" ;;
    unisum) INV="$MC/g7_unisum/units_gemma+ent_unisum.jsonl"; ACU="$MC/g7_unisum/acu_units_unisum.jsonl"; DEP="$MC/g7_unisum/val"; JT="$MCS/b3/val_unisum"; CMP="$MCS/hj3/eval_unisum/div" ;;
  esac
  for f in "$INV" "$ACU"; do [ -f "$f" ] || { echo "[hj9j] MISSING $f"; exit 4; }; done
  local SPECS="humanfact+div1=$INV:$(roster "$DEP" "gemma+ent_humanfact" "_div1.jsonl")"
  SPECS="$SPECS matched.judge+div1=$INV:$(roster "$JT" "gemma+ent_b3Ajudge" "_div1.jsonl")"
  for RB in hj9:factcg hj9_xlarge:xlarge; do
    local R="${RB%%:*}" BB="${RB##*:}"
    for LAB in human judge; do
      local ST="hj9${BB}${LAB}" V="$MCS/$R/val_${POOL}"
      [ -f "$V/gemma+ent_${ST}_seed20260906_div1.jsonl" ] || { echo "[hj9j] MISSING $ST on $POOL"; exit 4; }
      SPECS="$SPECS ${ST}+div1=$INV:$(roster "$V" "gemma+ent_${ST}" "_div1.jsonl")"
      SPECS="$SPECS ${ST}.shuffled+div1=$INV:$(roster "$V" "gemma+ent_${ST}shuf" "_div1.jsonl")"
      SPECS="$SPECS ${ST}.empty+div1=$INV:$(roster "$V" "gemma+ent_${ST}empty" "_div1.jsonl")"
    done
  done
  for f in "$CMP"/gemma+ent_nli_factcg_real_div1.jsonl "$CMP"/gemma+ent_nli_minicheck7b_real_div1.jsonl "$CMP"/gemma+ent_nli_judge_gemma31_real_div1.jsonl; do
    [ -e "$f" ] || { echo "[hj9j] MISSING comparator emission $f"; exit 4; }
    b=$(basename "$f" _div1.jsonl); SPECS="$SPECS ${b#gemma+ent_}+div1=$INV:$f"
  done
  echo "===== [hj9j] $POOL emitter set:"; for s in $SPECS; do echo "  ${s%%=*}"; done
  EV --acu "$ACU" --primary "humanfact+div1" --out "$OUT/eval_${POOL}" \
     --dump-per-pair "$OUT/per_pair_${POOL}_k10.jsonl" --dump-emitted "$OUT/emitted_${POOL}_k10.jsonl" \
     --systems $SPECS 2> >(tee -a "$OUT/e2_stderr.log" >&2) | grep -E "\[e2"
  "$PLAN_PY" "$SCODE/b3_contrast.py" --n-boot 2000 --boot-seed 20260918 --skip-missing \
    --eval "${POOL}_hj9joint=$OUT/eval_${POOL}:$OUT/per_pair_${POOL}_k10.jsonl" --out "$OUT/b3_contrasts.json"
  (cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
}
run_pool test
run_pool unisum
echo "[hj9j] done"
