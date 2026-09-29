#!/bin/bash
# M-A.2 -- one metric per array task over the meta-evaluation pairs. Depends on the M-A decomposition
# (mars-mm-decomp, MM_BLOCK=ma) for the MARS arms. Registered in research/mars_metric_20260923/PREREG.md.
#
#   sbatch --dependency=afterok:<decomp array> research/mars_metric_20260923/slurm/ma_score.sbatch
#SBATCH --job-name=mars-ma-score
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --array=0-8
#SBATCH --output=/home/user/results/mars_metric/logs/%x-%A_%a.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"
MM="${MARS_METRIC_OUT:-/home/user/results/mars_metric}"; CODE="$PLAN_REPO/research/mars_metric_20260923/code"
METRICS=(marsR_marsc marsP_factcg marsP_marsc marsR_factcg rouge bertscore bartscore alignscore factcg_whole)
M="${METRICS[${SLURM_ARRAY_TASK_ID:-0}]}"
PAIRS="$MM/ma/tasks/pairs_ma.jsonl"; [ -s "$PAIRS" ] || { echo "[ma-score] MISSING $PAIRS"; exit 4; }
mkdir -p "$MM/ma/scores"; plan_receipt "$MM/ma/scores/receipt_$M"
SNAP=$(snap "yaxili96/FactCG-DeBERTa-v3-Large") || true
BART=$(snap "facebook/bart-large-cnn") || true
RL=$(snap "FacebookAI/roberta-large") || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ARGS=(--pairs "$PAIRS" --metric "$M" --out "$MM/ma/scores/scores_$M.jsonl" --snapshot "$SNAP" --ckpts "$MC/e1/mc-A-lam1-crossenc-inv_seed*/model" --bart "$BART" --bertscore-model "$RL")
case "$M" in
  mars*) for t in 0 1; do [ -s "$MM/ma/decomp/props_shard$t.jsonl" ] || { echo "[ma-score] MISSING props shard $t"; exit 4; }; done
         ARGS+=(--props "$MM/ma/decomp/props_shard*.jsonl") ;;
esac
case "$M" in marsP_marsc|marsR_marsc) ARGS+=(--batch 128) ;; bertscore) ARGS+=(--batch 64) ;; *) ARGS+=(--batch 32) ;; esac
echo "[ma-score] metric=$M"
if [ "${MA_RAWLOG:-}" = "1" ]; then "$PLAN_PY" "$CODE/ma_score.py" "${ARGS[@]}" 2>&1 | tee "$MM/ma/scores/rawlog_$M.txt" | tail -n 40; else
"$PLAN_PY" "$CODE/ma_score.py" "${ARGS[@]}" 2>&1 | grep -E "^\[ma\]|^\[mb\] marsc|^\[b10\]|Error|Traceback" | grep -v "pairs (" || true; fi
[ -s "$MM/ma/scores/scores_$M.jsonl" ] || { echo "[ma-score] no output for $M"; exit 5; }
plan_done "$MM/ma/scores/receipt_$M"
echo "[ma-score] done -> $MM/ma/scores/scores_$M.jsonl"
