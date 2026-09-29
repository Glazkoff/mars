#!/bin/bash
# MARS-C R03 -- is the omission ranking conditioned on THIS summary, or a summary-blind prior over facts?
# (FULL_REVIEW_2026-09-17 decision-critical failure 4: "summary-conditioned omission detection is not isolated".)
#
#   R03a  within-document, same-fact crossed cells: P(z(f|summary that omits f) > z(f|summary that conveys f)).
#         A summary-blind scorer is pinned at exactly 0.500 here. Run on validation AND the sealed test split.
#   R03b  summary-shuffled emission: rescore the frozen TEST inventories with another pair's summary (seeded
#         derangement within resource x system), run the identical rule, evaluate recall@10 against the true labels.
#   R03c  fact-only floor: the same with no summary at all.
#
# R03b/c give the paper's own endpoint under the null: whatever recall survives with the wrong summary is what a
# summary-blind prior buys, and the real-summary recall minus that is what summary conditioning buys.
#SBATCH --job-name=mc-r03-sumblind
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[r03] REFUSED: MC_TEST_REREAD=yes not set"; exit 3; }
OUT="$MC/r03_sumblind"; mkdir -p "$OUT/scores" "$OUT/val"; plan_receipt "$OUT"
INV_GE="$M3/m32/units_gemma+ent_test.jsonl"; INV_DI="$M3/m32/units_distilled_test.jsonl"; ACU_T="$G/b18/acu_units_test.jsonl"
HFM="$MC/e1/mc-A-lam1-crossenc-inv_seed*/model"; JFM="$G/b21/xenc_seed*"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || { echo "[r03] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
EV () { "$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 "$@"; }

echo "===== R03a within-fact crossed cells (validation, then sealed TEST)"
for SPLIT in validation test; do
  "$PLAN_PY" "$MCCODE/mc_summary_conditioning.py" --docs "$MC/e0/docs_${SPLIT}.jsonl" \
    --verifiers "humanfact=$HFM" "judgefact=$JFM" --n-boot 2000 --out "$OUT/crossed_${SPLIT}" 2>&1 | grep -E "\[r03a"
done

echo "===== R03b/c rescoring the TEST inventories under shuffled / empty summaries"
for MODE in shuffled empty; do
  "$PLAN_PY" "$MCCODE/mc_score_units.py" --units "gemma+ent=$INV_GE" "distilled=$INV_DI" \
    --models "$HFM" --tag "humanfact$MODE" --candidate-mode "$MODE" --scores-out "$OUT/scores" 2>&1 | grep -E "\[score"
  "$PLAN_PY" "$M3CODE/m31_score_pipeline_all.py" --units "gemma+ent=$INV_GE" "distilled=$INV_DI" \
    --ckpts "$JFM" --tag "xenc$MODE" --candidate-mode "$MODE" --scores-out "$OUT/scores" 2>&1 | grep -E "\[pipe"
done
div () { "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "$1" --scores-glob "$2" --out "$OUT/val"; }
for MODE in shuffled empty; do
  div "gemma+ent=$INV_GE" "$OUT/scores/gemma+ent_humanfact${MODE}_seed*.jsonl"
  div "distilled=$INV_DI" "$OUT/scores/distilled_humanfact${MODE}_seed*.jsonl"
  div "gemma+ent=$INV_GE" "$OUT/scores/gemma+ent_xenc${MODE}_seed*.jsonl"
  div "distilled=$INV_DI" "$OUT/scores/distilled_xenc${MODE}_seed*.jsonl"
done

G1="$MC/g1_test"
SPECS="humanfact+div1=$INV_GE:$(roster "$G1/val" "gemma+ent_humanfact" "_div1.jsonl") \
judgefact.distilled+div1=$INV_DI:$(roster "$G1/val" "distilled_xenc" "_div1.jsonl")"
for MODE in shuffled empty; do
  SPECS="$SPECS humanfact.$MODE+div1=$INV_GE:$(roster "$OUT/val" "gemma+ent_humanfact${MODE}" "_div1.jsonl")"
  SPECS="$SPECS humanfact.distilled.$MODE+div1=$INV_DI:$(roster "$OUT/val" "distilled_humanfact${MODE}" "_div1.jsonl")"
  SPECS="$SPECS judgefact.distilled.$MODE+div1=$INV_DI:$(roster "$OUT/val" "distilled_xenc${MODE}" "_div1.jsonl")"
done
echo "===== R03b/c evaluation against the TRUE labels of the sealed TEST split"
EV --acu "$ACU_T" --k 5 10 --n-boot 500 --primary "humanfact+div1" --robust-tests --n-perm 10000 \
   --dump-per-pair "$OUT/per_pair_test.jsonl" --out "$OUT/eval_test" --systems $SPECS 2>&1 | grep -E "\[e2"
(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[r03] done -> $OUT"
