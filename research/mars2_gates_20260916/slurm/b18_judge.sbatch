#!/bin/bash
# B18: supplied-target Gemma judge on the human-labelled ACUs (validation), 8 shards. Usage: sbatch --dependency=afterok:<tasks> b18_judge.sbatch
#SBATCH --job-name=g-b18-judge
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/user/results/mars2_gates/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
SNAP="$(snap google/gemma-4-31B-it)"; [ -n "$SNAP" ] || { echo "no cached gemma snapshot"; exit 3; }
i="${SLURM_ARRAY_TASK_ID:-0}"; OUT="$G/b18/judge"; mkdir -p "$OUT"; plan_receipt "$OUT/receipt_$i"
"$PLAN_PY" scripts/plan2026/b1_judge_labels.py --tasks "$G/b18/judge_tasks_acu_${B18_SPLIT:-validation}.jsonl" \
  --out "$OUT/judge_shard${i}.jsonl" --model-snapshot "$SNAP" --shard "$i/8" --splits "${B18_SPLIT:-validation}" --batch-size 8
plan_done "$OUT/receipt_$i"
