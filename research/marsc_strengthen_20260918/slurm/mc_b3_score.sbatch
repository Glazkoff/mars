#!/bin/bash
# MARS-C B3 (step 1 of 2) -- EXTERNAL MATCHED-LABEL TRANSFER: scoring only, no retraining.
#
# What this closes.  The paper's only clean causal contrast -- label source, human vs judge, with unit
# construction, unit distribution, label budget, recipe, backbone, sampler, seeds and step budget all MATCHED
# -- exists on validation only (AUROC 0.9740 vs 0.9613). NUMBERS_PROVENANCE_2026-09-17 open item 5 reads
# "External matched-label (label-source) transfer not run." This job takes the twelve R04 checkpoints that
# already exist and scores the frozen gemma+ent inventories on the sealed TEST split and on UniSumEval.
#
# Nothing is trained. Nothing is tuned. PREREG.md declares before execution that on validation the label-source
# effect was about a fifth of the package effect (+0.023 of +0.122) and may shrink or reverse out of domain;
# that outcome is publishable and no knob in this job may be turned to avoid it.
#
# Grid: 8 checkpoint x candidate-mode configurations x 2 pools = 16 array tasks, one GPU each.
#   0 A-human real      4 A-human shuffled   (B0 control)      pool = task / 8:  0 = sealed TEST
#   1 A-judge real      5 A-human empty      (B0 control)                        1 = UniSumEval
#   2 R-human real      6 A-judge shuffled   (B0 control)
#   3 R-judge real      7 A-judge empty      (B0 control)
# Arm A is the deployed objective; arm R draws random natural cells, so its sampler cannot be influenced by the
# label-derived crossed structure -- R04 reported the two separately and so does B3.
# The B0 controls are attached to BOTH twins, not only the primary, because the campaign's identification
# instrument is the real-minus-blind gap and the label-source question is precisely whether that gap differs.
#
# Cost, from the measured 0.00083 GPU-s/unit verifier cost: TEST 300,360 units x 3 seeds = 12.5 GPU-min;
# UniSumEval 559,048 units x 3 seeds = 23 GPU-min at the same rate. UniSumEval summaries are long, so the
# 512-token premise is saturated far more often than on the RoSE news text the rate was measured on; the limit
# below carries that (a 2x slowdown plus three model loads plus a 91 MB inventory parse per seed still fits).
#SBATCH --job-name=mars-b3-score
#SBATCH --partition=train
#SBATCH --nodelist=Euler
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-15%2
#SBATCH --output=/home/user/results/marsc/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
# Both pools are sealed re-reads. PREREG.md (2026-09-18) is the dated amendment that authorises this.
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[b3] REFUSED: MC_TEST_REREAD=yes not set"; exit 3; }

B3="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}/b3"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
MODELS_ROOT="$MC/r04_matched/models"

TAGS=(b3Ahuman     b3Ajudge     b3Rhuman     b3Rjudge     b3Ahumanshuf b3Ahumanempty b3Ajudgeshuf b3Ajudgeempty)
FAMS=(A-r04human   A-r04judge   R-r04human   R-r04judge   A-r04human   A-r04human    A-r04judge   A-r04judge)
MODES=(real        real         real         real         shuffled     empty         shuffled     empty)
POOLS=(test unisum)

T="${SLURM_ARRAY_TASK_ID:?this sbatch must be submitted as an array}"
C=$(( T % 8 )); P=$(( T / 8 ))
TAG="${TAGS[$C]}"; FAM="${FAMS[$C]}"; MODE="${MODES[$C]}"; POOL="${POOLS[$P]}"
ARM="${FAM%%-*}"; LAB="${FAM##*-}"
PREFIX="mc-${ARM}-lam1-crossenc-${LAB}"

case "$POOL" in
  test)   INV="$M3/m32/units_gemma+ent_test.jsonl" ;;
  unisum) INV="$MC/g7_unisum/units_gemma+ent_unisum.jsonl" ;;
  *) echo "[b3] unknown pool $POOL" >&2; exit 2 ;;
esac
[ -f "$INV" ] || { echo "[b3] MISSING inventory $INV" >&2; exit 4; }

# The checkpoint roster is verified BEFORE any GPU work: a silently missing seed would otherwise produce a
# two-seed "three-seed" score set that --strict-seeds catches only hours later, in the evaluator.
for S in $SEEDS; do
  D="$MODELS_ROOT/${PREFIX}_seed${S}/model"
  [ -f "$D/config.json" ] || { echo "[b3] MISSING $D/config.json" >&2; exit 4; }
  [ -s "$D/model.safetensors" ] || { echo "[b3] MISSING OR EMPTY $D/model.safetensors" >&2; exit 4; }
done

SCORES="$B3/scores_${POOL}"; mkdir -p "$SCORES"
# One receipt per array task, in its own directory: 16 tasks writing one started.json would race,
# and the eval step writes the receipt for $B3 itself.
plan_receipt "$B3/receipts/score_task_${T}"
echo "[b3] task $T  pool=$POOL  arm=$ARM  labels=$LAB  mode=$MODE  tag=$TAG"
echo "[b3] inventory $INV"
echo "[b3] models    $MODELS_ROOT/${PREFIX}_seed*/model"

"$PLAN_PY" "$MCCODE/mc_score_units.py" \
  --units "gemma+ent=$INV" \
  --models "$MODELS_ROOT/${PREFIX}_seed*/model" \
  --tag "$TAG" --candidate-mode "$MODE" --scores-out "$SCORES"

# Post-condition: exactly the three registered seed files, non-empty.
for S in $SEEDS; do
  F="$SCORES/gemma+ent_${TAG}_seed${S}.jsonl"
  [ -s "$F" ] || { echo "[b3] task $T produced no $F" >&2; exit 5; }
  echo "[b3] wrote $F ($(wc -l < "$F") pairs)"
done
echo "[b3] task $T done"
