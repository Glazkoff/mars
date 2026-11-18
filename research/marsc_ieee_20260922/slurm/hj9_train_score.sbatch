#!/bin/bash
# H-J9 -- train the natural-coverage package on the FactCG-DeBERTa-v3-Large encoder with the deployed recipe,
# then score the frozen Gemma+ent inventories of the RoSE held-out split and UniSumEval in the three candidate
# modes and apply the frozen div1 rule. Registered in research/marsc_ieee_20260922/PREREG.md (amendment 6)
# BEFORE submission. One submission per label source:
#
#   sbatch --export=ALL,HJ9_LABEL=human,MC_TEST_REREAD=yes research/marsc_ieee_20260922/slurm/hj9_train_score.sbatch
#   sbatch --export=ALL,HJ9_LABEL=judge,MC_TEST_REREAD=yes research/marsc_ieee_20260922/slurm/hj9_train_score.sbatch
#SBATCH --job-name=mars-hj9-train
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=07:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[hj9] REFUSED: the held-out split is re-read; needs MC_TEST_REREAD=yes"; exit 3; }
LAB="${HJ9_LABEL:?set HJ9_LABEL=human|judge}"
BB="${HJ9_BACKBONE:-factcg}"
case "$BB" in
  factcg) BBSNAP=$(snap "yaxili96/FactCG-DeBERTa-v3-Large") || true ;;
  xlarge) BBSNAP=$(snap "microsoft/deberta-xlarge-mnli") || true ;;
  *) echo "[hj9] unknown HJ9_BACKBONE=$BB"; exit 2 ;;
esac
[ -n "$BBSNAP" ] && [ -f "$BBSNAP/config.json" ] || { echo "[hj9] MISSING backbone snapshot for $BB"; exit 4; }
case "$LAB" in
  human) DD="$MC/e0" ;;
  judge) DD="$MC/r04_matched/e0_judge" ;;
  *) echo "[hj9] unknown HJ9_LABEL=$LAB"; exit 2 ;;
esac
[ -f "$DD/docs_train.jsonl" ] || { echo "[hj9] MISSING $DD/docs_train.jsonl"; exit 4; }
HJ9="${HJ9_ROOT:-$MCS/hj9}"; mkdir -p "$HJ9/models" "$HJ9/scores_val" "$HJ9/scores_test" "$HJ9/scores_unisum" "$HJ9/val_test" "$HJ9/val_unisum" "$MCS/logs"
plan_receipt "$HJ9/receipt_${BB}_${LAB}"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
INV_T="$M3/m32/units_gemma+ent_test.jsonl"; INV_U="$MC/g7_unisum/units_gemma+ent_unisum.jsonl"
INV_V="$M3/m32/units_gemma+ent_validation.jsonl"; ACU_V="$G/b18/acu_units_validation.jsonl"
for f in "$INV_T" "$INV_U" "$INV_V" "$ACU_V"; do [ -f "$f" ] || { echo "[hj9] MISSING $f"; exit 4; }; done
TAG="mc-A-lam1-crossenc-hj9${BB}${LAB}"; STAG="hj9${BB}${LAB}"
echo "[hj9] backbone=$BB ($BBSNAP) label=$LAB data=$DD tag=$TAG"

echo "===== [hj9] train three seeds with the deployed recipe"
for S in $SEEDS; do
  MD="$HJ9/models/${TAG}_seed${S}"
  if [ -f "$MD/model/config.json" ]; then echo "[hj9] skip existing $TAG seed $S"; continue; fi
  mkdir -p "$MD"
  "$PLAN_PY" "$MCCODE/mc_train.py" --arm A --lam 1 --repr crossenc --seed "$S" --data-dir "$DD" \
    --backbone "$BBSNAP" --out "$MD" --tag "$TAG" --scores-out "$HJ9/scores_val" --eval-acu "$ACU_V" \
    --eval-units "units_gemma+ent=$INV_V" --save-model ${HJ9_EVAL_PAIRS:+--eval-pairs $HJ9_EVAL_PAIRS} ${HJ9_GRAD_CKPT:+--grad-ckpt} 2>&1 | grep -E "\[mc:|AUROC|auroc|error|Error" | tail -6 || true
  [ -f "$MD/model/config.json" ] || { echo "[hj9] training of seed $S produced no model"; exit 5; }
done

echo "===== [hj9] score both pools in the three candidate modes (explicit seed roster, three seeds)"
for POOL in test unisum; do
  case "$POOL" in test) INV="$INV_T" ;; unisum) INV="$INV_U" ;; esac
  for MODE in real shuffled empty; do
    case "$MODE" in real) T2="$STAG" ;; shuffled) T2="${STAG}shuf" ;; empty) T2="${STAG}empty" ;; esac
    "$PLAN_PY" "$MCCODE/mc_score_units.py" --units "gemma+ent=$INV" --models "$HJ9/models/${TAG}_seed*/model" \
      --tag "$T2" --candidate-mode "$MODE" --shuffle-seed 20260917 --scores-out "$HJ9/scores_${POOL}" 2>&1 | grep -E "\[score" || true
    for S in $SEEDS; do [ -s "$HJ9/scores_${POOL}/gemma+ent_${T2}_seed${S}.jsonl" ] || { echo "[hj9] missing score file for $T2 seed $S ($POOL)"; exit 5; }; done
  done
  echo "===== [hj9] the frozen div1 rule ($POOL)"
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "gemma+ent=$INV" --cap 1 \
    --scores-glob "$HJ9/scores_${POOL}/gemma+ent_${STAG}*_seed*.jsonl" --out "$HJ9/val_${POOL}"
  N=$(ls "$HJ9/val_${POOL}"/gemma+ent_${STAG}*_seed*_div1.jsonl | wc -l); [ "$N" -ge 9 ] || { echo "[hj9] expected 9 div1 files for $STAG on $POOL, found $N"; exit 5; }
done
(cd "$HJ9" && find models -name "result.json" | sort | xargs sha256sum > "SHA256SUMS_${BB}_${LAB}")
plan_done "$HJ9/receipt_${BB}_${LAB}"
echo "[hj9] done -> $HJ9 ($TAG)"
