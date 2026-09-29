#!/bin/bash
# H-J12 (audit response, 2026-09-28) -- the unit-matched judge twin: a judge-label verifier trained on Gemma+ent
# facts of the E0 train/calib documents (the deployed inventory's unit type), cell budget matched to R04.
# Registered in research/marsc_ieee_20260922/PREREG.md (Amendment H-J12) BEFORE submission.
#
# Stages (HJ12G_STAGE):
#   tasks  CPU: sample the units, write the judge tasks                       (minutes)
#   judge  GPU array 0-7: Gemma-4-31B-it coverage judge, b1_judge_labels.py (~25 min per shard, as R04b)
#   train  GPU: apply labels, train arm A x 3 seeds (the R04c recipe), score the RoSE TEST Gemma+ent and
#          distilled inventories and the UniSumEval Gemma+ent inventory, emission rule, frozen evaluator with
#          the R04 twins and the deployed systems in one call; then crossed accuracy on validation/TEST/UniSumEval
#
#   T=$(sbatch --parsable --export=ALL,HJ12G_STAGE=tasks --gres=none --cpus-per-task=4 --mem=32G --time=00:30:00 <this>)
#   J=$(sbatch --parsable --dependency=afterok:$T --array=0-7%4 --export=ALL,HJ12G_STAGE=judge --time=01:30:00 <this>)
#   sbatch --dependency=afterok:$J --export=ALL,HJ12G_STAGE=train,MC_TEST_REREAD=yes --time=06:00:00 <this>
#SBATCH --job-name=mars-hj12-gtwin
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A_%a.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"
MCCODE="$PLAN_REPO/research/marsc_20260916/code"; SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
JCODE="$PLAN_REPO/research/marsc_ieee_20260922/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
OUT="$MCS/hj12/gemma_twin"; mkdir -p "$OUT/judge" "$OUT/models" "$OUT/scores" "$OUT/val"
STAGE="${HJ12G_STAGE:?set HJ12G_STAGE=tasks|judge|train}"
SEEDS="20260906 20260907 20260908"

case "$STAGE" in
tasks)
  plan_receipt "$OUT/receipt_tasks"
  "$PLAN_PY" "$JCODE/hj12_gemma_twin.py" tasks --docs-dir "$MC/e0" --units "$M3/m32/train_rows_facts.jsonl" \
    --roles train,calib --seed 20260928 --out "$OUT"
  wc -l "$OUT/judge_tasks.jsonl"; plan_done "$OUT/receipt_tasks" ;;
judge)
  i="${SLURM_ARRAY_TASK_ID:-0}"; plan_receipt "$OUT/judge/receipt_$i"
  SNAP="$(snap google/gemma-4-31B-it)"; [ -n "$SNAP" ] || { echo "[hj12g] no cached gemma snapshot"; exit 3; }
  "$PLAN_PY" scripts/plan2026/b1_judge_labels.py --tasks "$OUT/judge_tasks.jsonl" \
    --out "$OUT/judge/judge_shard${i}.jsonl" --model-snapshot "$SNAP" --shard "$i/8" --splits train,calib --batch-size 8
  plan_done "$OUT/judge/receipt_$i" ;;
train)
  [ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[hj12g] REFUSED: MC_TEST_REREAD=yes not set"; exit 3; }
  plan_receipt "$OUT/receipt_train"
  "$PLAN_PY" "$JCODE/hj12_gemma_twin.py" apply --docs-dir "$MC/e0" --tasks-dir "$OUT" \
    --judge "$OUT/judge/judge_shard*.jsonl" --roles train,calib --out "$OUT/e0_gjudge"
  cp "$MC/e0/census.json" "$OUT/e0_gjudge/census.json"
  INV_GE="$M3/m32/units_gemma+ent_test.jsonl"; INV_DI="$M3/m32/units_distilled_test.jsonl"
  INV_GE_V="$M3/m32/units_gemma+ent_validation.jsonl"; ACU_T="$G/b18/acu_units_test.jsonl"; ACU_V="$G/b18/acu_units_validation.jsonl"
  TAG="mc-A-lam1-crossenc-gjudge"
  for S in $SEEDS; do
    MD="$OUT/models/${TAG}_seed${S}"; [ -f "$MD/model/config.json" ] && { echo "[hj12g] skip $S"; continue; }
    mkdir -p "$MD"
    "$PLAN_PY" "$MCCODE/mc_train.py" --arm A --lam 1 --repr crossenc --seed "$S" --data-dir "$OUT/e0_gjudge" \
      --out "$MD" --tag "$TAG" --scores-out "$OUT/scores" --eval-acu "$ACU_V" \
      --eval-units "units_gemma+ent=$INV_GE_V" --save-model 2>&1 | grep -E "\[mc:" | tail -4
  done
  "$PLAN_PY" "$MCCODE/mc_score_units.py" --units "gemma+ent=$INV_GE" "distilled=$INV_DI" \
    --models "$OUT/models/${TAG}_seed*/model" --tag gjudge --scores-out "$OUT/scores" 2>&1 | grep -E "\[score"
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "gemma+ent=$INV_GE" --scores-glob "$OUT/scores/gemma+ent_gjudge_seed*.jsonl" --out "$OUT/val"
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "distilled=$INV_DI" --scores-glob "$OUT/scores/distilled_gjudge_seed*.jsonl" --out "$OUT/val"
  roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || { echo "[hj12g] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
  R4="$MC/r04_matched"; G1="$MC/g1_test"
  "$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU_T" --k 10 --n-boot 2000 \
    --primary "deployed.humanfact+div1" --robust-tests --n-perm 10000 --out "$OUT/eval_test" --systems \
    "deployed.humanfact+div1=$INV_GE:$(roster "$G1/val" "gemma+ent_humanfact" "_div1.jsonl")" \
    "package.judge_spacy+div1=$INV_GE:$(roster "$G1/val" "gemma+ent_xenc" "_div1.jsonl")" \
    "twin.human_acu+div1=$INV_GE:$(roster "$R4/val_d" "gemma+ent_r04dhuman" "_div1.jsonl")" \
    "twin.judge_acu+div1=$INV_GE:$(roster "$R4/val_d" "gemma+ent_r04djudge" "_div1.jsonl")" \
    "twin.judge_gemma+div1=$INV_GE:$(roster "$OUT/val" "gemma+ent_gjudge" "_div1.jsonl")" 2>&1 | grep -E "^\[e2"
  "$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU_T" --k 10 --n-boot 2000 \
    --primary "deployed.humanfact.distilled+div1" --out "$OUT/eval_test_distilled" --systems \
    "deployed.humanfact.distilled+div1=$INV_DI:$(roster "$G1/val" "distilled_humanfact" "_div1.jsonl")" \
    "package.judge_spacy.distilled+div1=$INV_DI:$(roster "$G1/val" "distilled_xenc" "_div1.jsonl")" \
    "twin.judge_gemma.distilled+div1=$INV_DI:$(roster "$OUT/val" "distilled_gjudge" "_div1.jsonl")" 2>&1 | grep -E "^\[e2" || true
  echo "===== crossed accuracy with the new twin (caches of record copied)"
  UNI="$MCS/hj3/docs/docs_unisum.jsonl"
  for SP in validation test unisum; do
    case $SP in validation) D="$MC/e0/docs_validation.jsonl"; C="$MCS/b12/validation/cache";;
                test) D="$MC/e0/docs_test.jsonl"; C="$MCS/b12/test/cache";;
                unisum) D="$UNI"; C="$MCS/hj3/hcond_unisum/cache";; esac
    O="$OUT/crossed_$SP"; mkdir -p "$O/cache"; cp -n "$C/"*.json "$O/cache/"
    "$PLAN_PY" "$SCODE/b2_crossed_families.py" --docs "$D" --cache "$O/cache" --primary humanfact \
      --batch 64 --n-boot 2000 --seed 20260917 --expect-seeds 3 --out "$O" --families \
      "humanfact=ckpt:$MC/e1/mc-A-lam1-crossenc-inv_seed*/model" "judgefact=ckpt:$G/b21/xenc_seed*" \
      "gjudge=ckpt:$OUT/models/${TAG}_seed*/model" 2>&1 | grep -E "^\[b2\]"
  done
  (cd "$OUT" && find . -type f -name "*.json" ! -path "*/cache/*" | sort | xargs sha256sum > SHA256SUMS)
  plan_done "$OUT/receipt_train"; echo "[hj12g] done -> $OUT" ;;
*) echo "[hj12g] bad stage $STAGE"; exit 2 ;;
esac
