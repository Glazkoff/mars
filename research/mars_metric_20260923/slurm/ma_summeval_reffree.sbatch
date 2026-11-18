#!/bin/bash
# M-A.4 (gap fill, 2026-09-27) -- reference-free MARS-R on SummEval: source inventory (Gemma-4-31B decomposition,
# the frozen M32 prompt, plus spaCy entities), the deployed coverage verifier under the three candidate modes, the
# E2 importance tagger, then ma_reffree.py for the SummEval pairs into a separate score file that ma_meta.py folds in.
#SBATCH --job-name=mars-ma4-summeval
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=05:00:00
#SBATCH --output=/home/user/results/mars_metric/logs/%x-%j.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
MM="${MARS_METRIC_OUT:-/home/user/results/mars_metric}"; CODE="$PLAN_REPO/research/mars_metric_20260923/code"
OUT="$MM/ma/summeval_source"; mkdir -p "$OUT/decomp" "$OUT/scores" "$OUT/importance"; plan_receipt "$OUT"
"$PLAN_PY" "$CODE/ma_summeval_inventory.py" tasks --pairs "$MM/ma/tasks/pairs_ma.jsonl" --out "$OUT" \
  --train-sources "$MC/e0/docs_train.jsonl"
[ -s "$OUT/decomp/props_summeval.jsonl" ] || "$PLAN_PY" "$M3CODE/m32_decompose.py" --tasks "$OUT/decompose_tasks_summeval.jsonl" \
  --out "$OUT/decomp/props_summeval.jsonl" --shard 0/1 --batch-size 12 2>&1 | grep -E "\[m32" | tail -n 5
INV="$OUT/units_gemma+ent_summeval.jsonl"
"$PLAN_PY" "$CODE/ma_summeval_inventory.py" inventory --pairs "$OUT/pairs_summeval.jsonl" --props "$OUT/decomp/props_summeval.jsonl" --out "$INV"
for M in real shuffled empty; do
  T=humanfact; [ "$M" = real ] || T="humanfact${M}"
  "$PLAN_PY" "$MCCODE/mc_score_units.py" --units "gemma+ent=$INV" --models "$MC/e1/mc-A-lam1-crossenc-inv_seed*/model" \
    --tag "$T" --candidate-mode "$M" --shuffle-seed 20260917 --scores-out "$OUT/scores" --batch 64
done
"$PLAN_PY" "$MCCODE/mc_score_importance.py" --ckpts "$MC/e2imp_all/tagger_seed*" --units "gemma+ent=$INV" \
  --tag importance_all --scores-out "$OUT/importance" 2>&1 | grep -v Warning | tail -n 5
S="$OUT/scores"
"$PLAN_PY" "$CODE/ma_reffree.py" --pairs "$MM/ma/tasks/pairs_ma.jsonl" \
  --set "summeval=$S/gemma+ent_humanfact_seed*.jsonl|$OUT/importance/gemma+ent_importance_all_seed*.jsonl|$S/gemma+ent_humanfactshuffled_seed*.jsonl|$S/gemma+ent_humanfactempty_seed*.jsonl" \
  --out "$MM/ma/scores/scores_marsR_free_summeval.jsonl"
(cd "$OUT" && find . -maxdepth 2 -type f -name "*.json*" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[ma4] done"
