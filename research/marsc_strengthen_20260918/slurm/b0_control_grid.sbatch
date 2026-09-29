#!/bin/bash
# MARS-C strengthen B0 -- the standing control harness, one stage of one (split x inventory) cell.
#
# Every verifier family is scored on the IDENTICAL frozen inventory under all three registered candidate modes
# (real summary / seeded derangement within resource x system / no summary), pushed through the IDENTICAL div1
# emission rule, and evaluated with the IDENTICAL evaluator. The secondary estimands of B2 -- per family,
# real-minus-control on recall@10 AND on omission precision -- are computed from the evaluator's own per-pair
# dump by b0_control_gaps.py, because mc_e2_eval.py only contrasts against ONE primary and its
# `paired_recall.mean` is the BOOTSTRAP mean, not the observed contrast (the trap that retracted a claim here).
#
# The shuffled premise is produced by mc_score_units.premises for EVERY family (the zero-shot driver imports it),
# so the control column is the SAME wrong summary across families, not merely an analogous one.
#
# Two stages, so the GPU reservation stays short on a contended node and the analysis runs on the idle cores:
#   B0_STAGE=score  (GPU) one candidate mode, all families -> $OUT/scores
#   B0_STAGE=eval   (CPU) div1 + evaluator + control gaps over all three modes
#
#   sbatch --time=01:00:00 --export=ALL,B0_STAGE=score,B0_MODE=real,B0_SPLIT=validation,B0_INV=distilled <this>
#   sbatch --time=01:30:00 --partition=infer --gres=none --dependency=afterok:<3 score jobs> \
#          --export=ALL,B0_STAGE=eval,B0_SPLIT=validation,B0_INV=distilled <this>
# TEST additionally needs MC_TEST_REREAD=yes in --export.
#SBATCH --job-name=mars-b0-grid
#SBATCH --partition=train
#SBATCH --nodelist=Euler
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
SPLIT="${B0_SPLIT:-validation}"; INV="${B0_INV:-distilled}"; STAGE="${B0_STAGE:-score}"; MODE="${B0_MODE:-real}"
case "$SPLIT" in validation|test) ;; *) echo "[b0] bad B0_SPLIT=$SPLIT"; exit 2 ;; esac
case "$INV"   in distilled|gemma+ent) ;; *) echo "[b0] bad B0_INV=$INV"; exit 2 ;; esac
case "$STAGE" in score|eval) ;; *) echo "[b0] bad B0_STAGE=$STAGE"; exit 2 ;; esac
case "$MODE"  in real|shuffled|empty) ;; *) echo "[b0] bad B0_MODE=$MODE"; exit 2 ;; esac
if [ "$SPLIT" = "test" ]; then
  [ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[b0] REFUSED: sealed TEST re-read without MC_TEST_REREAD=yes"; exit 3; }
fi
INVFILE="$M3/m32/units_${INV}_${SPLIT}.jsonl"; ACU="$G/b18/acu_units_${SPLIT}.jsonl"
for f in "$INVFILE" "$ACU"; do [ -f "$f" ] || { echo "[b0] MISSING $f"; exit 4; }; done
HFM="$MC/e1/mc-A-lam1-crossenc-inv_seed*/model"; JFM="$G/b21/xenc_seed*"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
OUT="$MCS/b0/${SPLIT}_${INV}"; mkdir -p "$OUT/scores" "$OUT/val"
plan_receipt "$OUT/receipts_${STAGE}_${MODE}"
echo "[b0] stage=$STAGE split=$SPLIT inventory=$INV mode=$MODE units=$INVFILE acu=$ACU out=$OUT"

if [ "$STAGE" = "score" ]; then
  echo "===== candidate-mode=$MODE : MARS-C humanfact cross-encoder (3 seeds)"
  "$PLAN_PY" "$MCCODE/mc_score_units.py" --units "${INV}=$INVFILE" --models "$HFM" \
    --tag "humanfact-${MODE}" --candidate-mode "$MODE" --shuffle-seed 20260917 \
    --scores-out "$OUT/scores" --batch 64
  echo "===== candidate-mode=$MODE : MARS-C judge-label xenc (3 seeds)"
  "$PLAN_PY" "$M3CODE/m31_score_pipeline_all.py" --units "${INV}=$INVFILE" --ckpts "$JFM" \
    --tag "judgefact-${MODE}" --candidate-mode "$MODE" --shuffle-seed 20260917 \
    --scores-out "$OUT/scores" --batch 64
  echo "===== candidate-mode=$MODE : off-the-shelf families"
  "$PLAN_PY" "$SCODE/b0_zs_score_units.py" --units "${INV}=$INVFILE" \
    --systems lex,minilm,nli,summac_zs --candidate-mode "$MODE" --shuffle-seed 20260917 \
    --scores-out "$OUT/scores" --batch 32
  echo "[b0] score stage done -> $OUT/scores (mode=$MODE)"
  exit 0
fi

echo "===== stage eval.1: the identical div1 emission rule for every arm"
"$PLAN_PY" "$MCCODE/mc_diversity.py" --units "${INV}=$INVFILE" --scores-glob "$OUT/scores/*.jsonl" \
  --out "$OUT/val" --cap 1

echo "===== stage eval.2: the evaluator, explicit three-seed manifests only (never a glob)"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || { echo "[b0] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
one () { local f="$1"; [ -f "$f" ] || { echo "[b0] MISSING $f" >&2; exit 4; }; printf '%s' "$f"; }
SPECS=""
for M in real shuffled empty; do
  SPECS="$SPECS humanfact.${INV}@${M}=$INVFILE:$(roster "$OUT/val" "${INV}_humanfact-${M}" "_div1.jsonl")"
  SPECS="$SPECS judgefact.${INV}@${M}=$INVFILE:$(roster "$OUT/val" "${INV}_judgefact-${M}" "_div1.jsonl")"
  for Z in lex_token_recall lex_rougeL_recall emb_minilm_maxcos nli_nli nli_summac_zs; do
    SPECS="$SPECS ${Z}.${INV}@${M}=$INVFILE:$(one "$OUT/val/${INV}_${Z}_${M}_div1.jsonl")"
  done
done
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU" --k 10 --n-boot 500 \
  --primary "humanfact.${INV}@real" --dump-per-pair "$OUT/per_pair_k10.jsonl" --out "$OUT/eval" \
  --systems $SPECS

echo "===== stage eval.3: the within-family control gaps (the B2 secondary estimands)"
"$PLAN_PY" "$SCODE/b0_control_gaps.py" --per-pair "$OUT/per_pair_k10.jsonl" --k 10 \
  --modes shuffled,empty --reference real --sep "@" --n-boot 2000 --out "$OUT"

(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[b0] done -> $OUT"
