#!/bin/bash
# MARS-C R04c -- the matched-unit, matched-count label-source control (FULL_REVIEW_2026-09-17 failure 3).
#
# Two twins of the E0 training data that differ ONLY in who produced the coverage label of each cell:
#   human  $MC/e0                (RoSE annotators)
#   judge  $MC/r04_matched/e0_judge  (Gemma-4-31B-it, same prompt as the paper's judge pipeline, same cells)
# Identical documents, identical non-reference summaries, identical human ACUs as fact units, identical
# recipe/backbone/sampler/seeds/step budget. Arm A is the deployed objective; arm R (random natural cells, no
# tetrads) is run as well because its sampler cannot be influenced by the label-derived crossed structure.
#
# The gap this measures IS the label source. The paper's headline gap additionally carries the unit
# construction, the unit distribution and a 6x larger label budget; the two numbers are reported side by side.
#SBATCH --job-name=mc-r04c-matched
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
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[r04c] REFUSED: MC_TEST_REREAD=yes not set"; exit 3; }
OUT="$MC/r04_matched"; mkdir -p "$OUT/scores" "$OUT/val" "$OUT/models"; plan_receipt "$OUT"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
INV_GE="$M3/m32/units_gemma+ent_test.jsonl"; INV_DI="$M3/m32/units_distilled_test.jsonl"
INV_GE_V="$M3/m32/units_gemma+ent_validation.jsonl"; ACU_T="$G/b18/acu_units_test.jsonl"; ACU_V="$G/b18/acu_units_validation.jsonl"

echo "===== R04c.0 apply the judge labels to a twin of the E0 data"
"$PLAN_PY" "$MCCODE/mc_r04_relabel.py" apply --docs-dir "$MC/e0" --judge "$OUT/judge/judge_shard*.jsonl" \
  --roles train,calib --out "$OUT/e0_judge"
cp "$MC/e0/census.json" "$OUT/e0_judge/census.json"          # gate record travels with the twin (E0 gate is label-agnostic)

echo "===== R04c.1 train both twins, both arms, three seeds"
for LAB in human judge; do
  [ "$LAB" = human ] && DD="$MC/e0" || DD="$OUT/e0_judge"
  for ARM in A R; do
    for S in $SEEDS; do
      TAG="mc-${ARM}-lam1-crossenc-r04${LAB}"; MD="$OUT/models/${TAG}_seed${S}"
      [ -f "$MD/model/config.json" ] && { echo "[r04c] skip existing $TAG seed $S"; continue; }
      mkdir -p "$MD"
      "$PLAN_PY" "$MCCODE/mc_train.py" --arm "$ARM" --lam 1 --repr crossenc --seed "$S" --data-dir "$DD" \
        --out "$MD" --tag "$TAG" --scores-out "$OUT/scores" --eval-acu "$ACU_V" \
        --eval-units "units_gemma+ent=$INV_GE_V" --save-model 2>&1 | grep -E "\[mc:" | tail -4
    done
  done
done

echo "===== R04c.2 score the sealed TEST inventories with every matched verifier"
for LAB in human judge; do for ARM in A R; do
  TAG="mc-${ARM}-lam1-crossenc-r04${LAB}"
  "$PLAN_PY" "$MCCODE/mc_score_units.py" --units "gemma+ent=$INV_GE" "distilled=$INV_DI" \
    --models "$OUT/models/${TAG}_seed*/model" --tag "r04${ARM}${LAB}" --scores-out "$OUT/scores" 2>&1 | grep -E "\[score"
done; done

echo "===== R04c.3 the emission rule, then the frozen TEST protocol"
div () { "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "$1" --scores-glob "$2" --out "$OUT/val"; }
for LAB in human judge; do for ARM in A R; do
  div "gemma+ent=$INV_GE" "$OUT/scores/gemma+ent_r04${ARM}${LAB}_seed*.jsonl"
  div "distilled=$INV_DI" "$OUT/scores/distilled_r04${ARM}${LAB}_seed*.jsonl"
done; done
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || { echo "[r04c] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
SPECS=""
for LAB in human judge; do for ARM in A R; do
  SPECS="$SPECS matched.${ARM}.${LAB}+div1=$INV_GE:$(roster "$OUT/val" "gemma+ent_r04${ARM}${LAB}" "_div1.jsonl")"
  SPECS="$SPECS matched.${ARM}.${LAB}.distilled+div1=$INV_DI:$(roster "$OUT/val" "distilled_r04${ARM}${LAB}" "_div1.jsonl")"
done; done
G1="$MC/g1_test"
SPECS="$SPECS deployed.humanfact+div1=$INV_GE:$(roster "$G1/val" "gemma+ent_humanfact" "_div1.jsonl")"
SPECS="$SPECS deployed.judgefact+div1=$INV_GE:$(roster "$G1/val" "gemma+ent_xenc" "_div1.jsonl")"
SPECS="$SPECS deployed.judgefact.distilled+div1=$INV_DI:$(roster "$G1/val" "distilled_xenc" "_div1.jsonl")"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU_T" --k 5 10 --n-boot 500 \
  --primary "matched.A.human+div1" --robust-tests --n-perm 10000 --out "$OUT/eval_test" --systems $SPECS 2>&1 | grep -E "\[e2"

echo "===== R04c.4 supplied-ACU AUROC of every matched verifier on validation"
"$PLAN_PY" "$MCCODE/mc_analyze.py" --data-dir "$MC/e0" --scores-glob "$OUT/scores/acu_mc-*-r04*_seed*.jsonl" --out "$OUT/analysis" 2>&1 | tail -30
(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[r04c] done -> $OUT"
