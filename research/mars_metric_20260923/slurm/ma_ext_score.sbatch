#!/bin/bash
# M-A.3 -- additional coverage and multi-dimension comparators over the M-A pairs (Amendment M-A.3 in PREREG.md).
# One array task per (metric, shard), one H200 each. QAEval runs in ~/.venv-qaeval-gpu: qaeval 0.1.0 with its
# transformers 3.0.2 / allennlp stack and torch 2.4 (torch 1.6, its pin, has no H200 kernels).
#   task 0 unieval | 1 a3cu | 2 a2cu | 3-6 questeval shards 0-3 | 7-10 qaeval shards 0-3
#SBATCH --job-name=mars-ma-ext
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-10
#SBATCH --output=/home/user/results/mars_metric/logs/%x-%A_%a.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
MM="${MARS_METRIC_OUT:-/home/user/results/mars_metric}"; CODE="$PLAN_REPO/research/mars_metric_20260923/code"
PAIRS="$MM/ma/tasks/pairs_ma.jsonl"; [ -s "$PAIRS" ] || { echo "[ma-ext] MISSING $PAIRS"; exit 4; }
T=${SLURM_ARRAY_TASK_ID:-0}
if   [ "$T" -eq 0 ]; then M=unieval; SH=0/1
elif [ "$T" -eq 1 ]; then M=a3cu; SH=0/1
elif [ "$T" -eq 2 ]; then M=a2cu; SH=0/1
elif [ "$T" -le 6 ]; then M=questeval; SH=$((T-3))/4
else M=qaeval; SH=$((T-7))/4; fi
TAG="$M"; [ "$SH" = "0/1" ] || TAG="${M}_s$(printf %02d ${SH%/*})"
OUT="$MM/ma/scores/scores_$TAG.jsonl"
mkdir -p "$MM/ma/scores"; plan_receipt "$MM/ma/scores/receipt_$TAG"
export HF_HOME=/home/user/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
UE=$(ls -d /home/user/hf_cache/hub/models--MingZhong--unieval-sum/snapshots/*/ | head -1)
echo "[ma-ext] metric=$M shard=$SH -> $OUT"
"$PLAN_PY" "$CODE/ma_ext_score.py" --pairs "$PAIRS" --metric "$M" --shard "$SH" --out "$OUT" --unieval "$UE" --batch 32
[ -s "$OUT" ] || { echo "[ma-ext] no output for $TAG"; exit 5; }
plan_done "$MM/ma/scores/receipt_$TAG"
echo "[ma-ext] done -> $OUT"
