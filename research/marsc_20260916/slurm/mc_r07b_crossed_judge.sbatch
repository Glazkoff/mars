#!/bin/bash
# MARS-C R07b -- the validated coverage judge over the crossed design's emitted units (top 10 only).
#SBATCH --job-name=mc-r07b-judge
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/home/user/results/marsc/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
OUT="$MC/r07_crossed"; mkdir -p "$OUT/judgments"
T="${SLURM_ARRAY_TASK_ID:-0}"
SNAP=$("$PLAN_PY" -c "import sys; sys.path.insert(0,'$PLAN_REPO/research/mars3_20260916/code'); import m3common as m; print(m.snap('google/gemma-4-31B-it'))")
[ -n "$SNAP" ] || { echo "[r07b] no gemma snapshot"; exit 3; }
plan_receipt "$OUT/judgments/receipt_$T"
"$PLAN_PY" "$MCCODE/mc_judge_precision.py" run --tasks "$OUT/tasks.jsonl" \
  --out "$OUT/judgments/gemma_coverage_shard${T}.jsonl" --model-snapshot "$SNAP" --question coverage --shard "$T/8" --batch-size 8
plan_done "$OUT/judgments/receipt_$T"
