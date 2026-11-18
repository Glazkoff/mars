#!/bin/bash
# M-E -- end-to-end cost of the package, one arm per array task, three repeats each (the median of repeats is
# reported). To be submitted only when M-E is opened (PREREG.md, queued designs).
#
#   sbatch research/mars_metric_20260923/slurm/me_cost.sbatch
#SBATCH --job-name=mars-me-cost
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/user/results/mars_metric/logs/%x-%A_%a.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MM="${MARS_METRIC_OUT:-/home/user/results/mars_metric}"
CODE="$PLAN_REPO/research/mars_metric_20260923/code"; ARMS=(mars_seq2seq mars_llm mars_sentences factcg_sent); ARM="${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"
FACTCG=$(snap "yaxili96/FactCG-DeBERTa-v3-Large"); GEMMA=$(snap "google/gemma-4-31B-it")
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PLAN_REPO"
mkdir -p "$MM/me"; plan_receipt "$MM/me/receipt_$ARM"
for REP in 1 2 3; do
  "$PLAN_PY" "$CODE/me_cost.py" --pairs "$MM/ma/tasks/pairs_ma.jsonl" --arm "$ARM" --factcg "$FACTCG" --marsc "$MC/e1/mc-A-lam1-crossenc-inv_seed*/model" \
    --segmenter /home/user/results/mars3/m32/segmenter --llm "$GEMMA" --out "$MM/me/cost_${ARM}_rep$REP.json"
done
plan_done "$MM/me/receipt_$ARM"; echo "[me] done $ARM"
