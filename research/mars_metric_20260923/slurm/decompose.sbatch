#!/bin/bash
# MARS metric program -- Gemma-4-31B sentence decomposition (the frozen M32 recipe and prompt) of a task file,
# two shards. Used by M-A.1 (meta-evaluation candidates and references) and M-B.1 (LLM-AggreFact claims).
# Registered in research/mars_metric_20260923/PREREG.md before submission.
#
#   sbatch --export=ALL,MM_BLOCK=mb research/mars_metric_20260923/slurm/decompose.sbatch
#   sbatch --export=ALL,MM_BLOCK=ma research/mars_metric_20260923/slurm/decompose.sbatch
#SBATCH --job-name=mars-mm-decomp
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/user/results/mars_metric/logs/%x-%A_%a.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
MM="${MARS_METRIC_OUT:-/home/user/results/mars_metric}"; B="${MM_BLOCK:?set MM_BLOCK=ma|mb}"; T="${SLURM_ARRAY_TASK_ID:-0}"
TASKS="$MM/$B/tasks/tasks_$B.jsonl"; [ -s "$TASKS" ] || { echo "[mm-decomp] MISSING $TASKS (build it on the login node first)"; exit 4; }
mkdir -p "$MM/$B/decomp" "$MM/logs"
echo "[mm-decomp] block=$B shard=$T/2 tasks=$(grep -c . "$TASKS")"
"$PLAN_PY" "$M3CODE/m32_decompose.py" --tasks "$TASKS" --out "$MM/$B/decomp/props_shard$T.jsonl" --shard "$T/2" --batch-size 12 2>&1 | grep -E "\[m32" | tail -n 5
N=$(grep -c . "$MM/$B/decomp/props_shard$T.jsonl"); echo "[mm-decomp] shard $T wrote $N tasks"
[ "$N" -gt 0 ] || { echo "[mm-decomp] empty output"; exit 5; }
echo "[mm-decomp] done"
