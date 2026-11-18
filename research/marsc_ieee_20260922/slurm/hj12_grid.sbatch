#!/bin/bash
# H-J12 (audit response, 2026-09-28) -- the empty cells of the inventory x verifier grid (review finding F07).
# Registered in research/marsc_ieee_20260922/PREREG.md (Amendment H-J12, part d) BEFORE submission.
#
#   HJ12_STAGE=twin  GPU: the R04 label-only twins (human and judge) on the distilled inventory of RoSE TEST and of
#                    UniSumEval, frozen emission rule. The three comparators on distilled UniSumEval are separate
#                    submissions of hj3_score_unisum.sbatch (HJ3_INV=distilled, real mode).
#   HJ12_STAGE=eval  CPU: one evaluator call per pool on the distilled inventory with every verifier in the grid,
#                    primary = the deployed human-fact verifier; observed contrasts with document-clustered intervals.
#SBATCH --job-name=mars-hj12-grid
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%j.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[hj12grid] REFUSED: MC_TEST_REREAD=yes not set"; exit 3; }
OUT="$MCS/hj12/grid"; mkdir -p "$OUT"
SEEDS="20260906 20260907 20260908"
INV_DT="$M3/m32/units_distilled_test.jsonl"; ACU_T="$G/b18/acu_units_test.jsonl"
D="$MCS/hj11/unisum_distilled"; INV_DU="$D/units_distilled_unisum.jsonl"; ACU_U="$MC/g7_unisum/acu_units_unisum.jsonl"
R4="$MC/r04_matched/models"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || { echo "[hj12grid] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }

case "${HJ12_STAGE:?set HJ12_STAGE=twin|eval}" in
twin)
  plan_receipt "$OUT/twin"
  for POOL in test unisum; do
    [ $POOL = test ] && INV="$INV_DT" || INV="$INV_DU"
    mkdir -p "$OUT/$POOL/scores" "$OUT/$POOL/val"
    for LAB in human judge; do
      "$PLAN_PY" "$MCCODE/mc_score_units.py" --units "distilled=$INV" \
        --models "$R4/mc-A-lam1-crossenc-r04${LAB}_seed*/model" --tag "r04${LAB}" --scores-out "$OUT/$POOL/scores" 2>&1 | grep -E "\[score"
      "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "distilled=$INV" \
        --scores-glob "$OUT/$POOL/scores/distilled_r04${LAB}_seed*.jsonl" --out "$OUT/$POOL/val"
    done
  done
  plan_done "$OUT/twin" ;;
eval)
  plan_receipt "$OUT/eval"
  echo "===== RoSE TEST, distilled inventory"
  G1="$MC/g1_test"; CT="$MCS/b9b10_eval/test/div"
  SPECS="humanfact.distilled+div1=$INV_DT:$(roster "$G1/val" distilled_humanfact _div1.jsonl)"
  SPECS="$SPECS package.distilled+div1=$INV_DT:$(roster "$G1/val" distilled_xenc _div1.jsonl)"
  SPECS="$SPECS twinhuman.distilled+div1=$INV_DT:$(roster "$OUT/test/val" distilled_r04human _div1.jsonl)"
  SPECS="$SPECS twinjudge.distilled+div1=$INV_DT:$(roster "$OUT/test/val" distilled_r04judge _div1.jsonl)"
  for f in "$CT"/distilled_*real*_div1.jsonl; do [ -e "$f" ] || continue; b=$(basename "$f" _div1.jsonl); SPECS="$SPECS ${b#distilled_}.distilled+div1=$INV_DT:$f"; done
  echo "emitters:"; for s in $SPECS; do echo "  ${s%%=*}"; done
  "$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU_T" --k 10 --n-boot 2000 \
    --primary "humanfact.distilled+div1" --out "$OUT/eval/eval_test" --dump-per-pair "$OUT/eval/per_pair_test_k10.jsonl" \
    --systems $SPECS 2>&1 | grep -E "\[e2"
  echo "===== UniSumEval, distilled inventory"
  CU="$OUT/unisum/comparators"; mkdir -p "$CU"
  for SYS in judge factcg minicheck7b; do
    SRC="$MCS/hj12/unisum_distilled_scores/$SYS/scores"
    ls "$SRC"/distilled_*real*.jsonl >/dev/null 2>&1 || { echo "[hj12grid] MISSING $SYS distilled scores"; exit 4; }
    "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "distilled=$INV_DU" --scores-glob "$SRC/distilled_*real*.jsonl" --out "$CU"
  done
  SPECS="humanfact.distilled+div1=$INV_DU:$(roster "$D/val" distilled_humanfact _div1.jsonl)"
  SPECS="$SPECS package.distilled+div1=$INV_DU:$(roster "$D/val" distilled_xenc _div1.jsonl)"
  SPECS="$SPECS twinhuman.distilled+div1=$INV_DU:$(roster "$OUT/unisum/val" distilled_r04human _div1.jsonl)"
  SPECS="$SPECS twinjudge.distilled+div1=$INV_DU:$(roster "$OUT/unisum/val" distilled_r04judge _div1.jsonl)"
  for f in "$CU"/distilled_*_div1.jsonl; do [ -e "$f" ] || continue; b=$(basename "$f" _div1.jsonl); SPECS="$SPECS ${b#distilled_}.distilled+div1=$INV_DU:$f"; done
  echo "emitters:"; for s in $SPECS; do echo "  ${s%%=*}"; done
  "$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU_U" --k 10 --n-boot 2000 \
    --primary "humanfact.distilled+div1" --out "$OUT/eval/eval_unisum" --dump-per-pair "$OUT/eval/per_pair_unisum_k10.jsonl" \
    --systems $SPECS 2>&1 | grep -E "\[e2"
  "$PLAN_PY" "$SCODE/b3_contrast.py" --n-boot 2000 --boot-seed 20260918 --skip-missing \
    --eval "test_distilled=$OUT/eval/eval_test:$OUT/eval/per_pair_test_k10.jsonl" \
           "unisum_distilled=$OUT/eval/eval_unisum:$OUT/eval/per_pair_unisum_k10.jsonl" --out "$OUT/eval/b3_contrasts.json"
  (cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT/eval" ;;
*) echo "[hj12grid] bad stage"; exit 2 ;;
esac
echo "[hj12grid] done -> $OUT"
