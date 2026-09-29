#!/bin/bash
# MARS-C strengthening B4c -- score the unseen extractor's inventory with the FROZEN verifiers.
#
# Nothing is retrained and no threshold is refitted: the primary MARS-C cross-encoder (E1 arm A, 3 seeds) and
# the judge-label verifier (B21 xenc, 3 seeds) are applied to the new inventory exactly as they are applied to
# the development ones, through the same two scorers the development results were produced with
# (mc_score_units.py for the 1-logit human-fact head, m31_score_pipeline_all.py for the 2-logit judge head).
#
# Every arm is produced three times -- real summary, seeded within-stratum shuffled summary, no summary at all
# -- because the campaign's standing rule (PREREG Block B0) forbids publishing any arm without its
# summary-blind controls: recall@k rewards ignoring the candidate, so an arm that survives only on recall is
# not evidence of anything.
#
# Array task 0 = the human-fact verifier, task 1 = the judge-label verifier; two GPUs, half the wall clock.
#SBATCH --job-name=mars-b4c-score
# PARTITION NOTE: GPU work runs on `infer`, not `train`. Both partitions serve the same eight H200s on
# the same node, `infer` carries PriorityTier=100 against train's 1, and every existing MARS-C GPU job
# (mc_e1, mc_g1_test, mc_g7b) was submitted there. At submission `train` held eighteen pending GPU jobs
# and a nine-hour start estimate. Nothing is cancelled, requeued or preempted by this choice.
#SBATCH --partition=infer
#SBATCH --nodelist=Euler
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A_%a.out
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[b4c] REFUSED: MC_TEST_REREAD=yes not set (sealed TEST split)"; exit 3; }
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
OUT="${B4_OUT:-/home/user/results/marsc_strengthen/b4}"; T="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$OUT/scores"
INV="$OUT/inv/units_qwenprops+ent_test.jsonl"
[ -s "$INV" ] || { echo "[b4c] MISSING or empty $INV"; exit 4; }
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
for s in $SEEDS; do
  [ -f "$MC/e1/mc-A-lam1-crossenc-inv_seed$s/model/config.json" ] || { echo "[b4c] MISSING humanfact seed $s"; exit 4; }
  [ -f "$G/b21/xenc_seed$s/config.json" ] || { echo "[b4c] MISSING xenc seed $s"; exit 4; }
done

if [ "$T" = "0" ]; then
  echo "===== B4c.0 the primary MARS-C human-fact cross-encoder, 3 seeds x {real, shuffled, empty}"
  "$PLAN_PY" "$MCCODE/mc_score_units.py" --units "qwenprops+ent=$INV" \
    --models "$MC/e1/mc-A-lam1-crossenc-inv_seed*/model" --tag humanfact \
    --candidate-mode real --scores-out "$OUT/scores"
  "$PLAN_PY" "$MCCODE/mc_score_units.py" --units "qwenprops+ent=$INV" \
    --models "$MC/e1/mc-A-lam1-crossenc-inv_seed*/model" --tag humanfactshuffled \
    --candidate-mode shuffled --shuffle-seed 20260917 --scores-out "$OUT/scores"
  "$PLAN_PY" "$MCCODE/mc_score_units.py" --units "qwenprops+ent=$INV" \
    --models "$MC/e1/mc-A-lam1-crossenc-inv_seed*/model" --tag humanfactempty \
    --candidate-mode empty --scores-out "$OUT/scores"
else
  echo "===== B4c.1 the judge-label verifier, 3 seeds x {real, shuffled, empty}"
  "$PLAN_PY" "$M3CODE/m31_score_pipeline_all.py" --units "qwenprops+ent=$INV" \
    --ckpts "$G/b21/xenc_seed*" --tag xenc --candidate-mode real --scores-out "$OUT/scores"
  "$PLAN_PY" "$M3CODE/m31_score_pipeline_all.py" --units "qwenprops+ent=$INV" \
    --ckpts "$G/b21/xenc_seed*" --tag xencshuffled --candidate-mode shuffled --shuffle-seed 20260917 --scores-out "$OUT/scores"
  "$PLAN_PY" "$M3CODE/m31_score_pipeline_all.py" --units "qwenprops+ent=$INV" \
    --ckpts "$G/b21/xenc_seed*" --tag xencempty --candidate-mode empty --scores-out "$OUT/scores"
fi
echo "[b4c] task $T done"
