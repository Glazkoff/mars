#!/bin/bash
# M-B.2 -- score LLM-AggreFact (one split per array task) with one verifier, whole claim and decomposed, plus
# the shuffled-document control on test. Depends on the M-B decomposition (mars-mm-decomp, MM_BLOCK=mb).
# Registered in research/mars_metric_20260923/PREREG.md.
#
#   sbatch --dependency=afterok:<decomp array> --export=ALL,MB_VERIFIER=factcg research/mars_metric_20260923/slurm/mb_score.sbatch
#   sbatch --dependency=afterok:<decomp array> --export=ALL,MB_VERIFIER=marsc  research/mars_metric_20260923/slurm/mb_score.sbatch
#SBATCH --job-name=mars-mb-score
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/user/results/mars_metric/logs/%x-%A_%a.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"
MM="${MARS_METRIC_OUT:-/home/user/results/mars_metric}"; CODE="$PLAN_REPO/research/mars_metric_20260923/code"
V="${MB_VERIFIER:?set MB_VERIFIER=factcg|marsc}"; T="${SLURM_ARRAY_TASK_ID:-0}"; SPLIT=$([ "$T" -lt 4 ] && echo dev || echo test); SH=$((T % 4))
AGG=/home/user/hf_cache/hub/datasets--lytang--LLM-AggreFact/snapshots/981dfd0bd8e58e7238a9ab92b2e6ea44bce918e4/data
for t in 0 1; do [ -s "$MM/mb/decomp/props_shard$t.jsonl" ] || { echo "[mb-score] MISSING props shard $t"; exit 4; }; done
mkdir -p "$MM/mb/scores"; plan_receipt "$MM/mb/scores/receipt_${V}_${SPLIT}_shard${SH}"
case "$V" in
  factcg) SNAP=$(snap "yaxili96/FactCG-DeBERTa-v3-Large") || true; [ -f "$SNAP/config.json" ] || { echo "[mb-score] MISSING FactCG snapshot"; exit 4; }
          ARGS=(--verifier factcg --snapshot "$SNAP" --batch 32) ;;
  marsc)  ARGS=(--verifier marsc --ckpts "$MC/e1/mc-A-lam1-crossenc-inv_seed*/model" --batch 128) ;;
  *) echo "[mb-score] unknown MB_VERIFIER=$V"; exit 2 ;;
esac
echo "[mb-score] verifier=$V split=$SPLIT shard=$SH/4"
"$PLAN_PY" "$CODE/mb_score.py" --aggrefact-dir "$AGG" --props "$MM/mb/decomp/props_shard*.jsonl" --split "$SPLIT" --shard "$SH/4" "${ARGS[@]}" \
  --out "$MM/mb/scores/scores_${V}_${SPLIT}_shard${SH}.jsonl" 2>&1 | grep -E "^\[mb\]|^\[b10\]|Error|Traceback" | grep -v "pairs (" || true
[ -s "$MM/mb/scores/scores_${V}_${SPLIT}_shard${SH}.jsonl" ] || { echo "[mb-score] no output"; exit 5; }
plan_done "$MM/mb/scores/receipt_${V}_${SPLIT}_shard${SH}"
echo "[mb-score] done -> $MM/mb/scores/scores_${V}_${SPLIT}_shard${SH}.jsonl"
