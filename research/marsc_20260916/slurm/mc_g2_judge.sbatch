#!/bin/bash
# MARS-C G2 -- judge-estimated precision of emitted facts: two families x three shards; each shard answers both questions.
#SBATCH --job-name=mc-g2-judge
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/user/results/marsc/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"; G2="$MC/g2"; mkdir -p "$G2/judgments"
T="${SLURM_ARRAY_TASK_ID:-0}"; FAMS=(gemma qwen); MODELS=("google/gemma-4-31B-it" "Qwen/Qwen3.8-27B"); FAM="${FAMS[$((T / 3))]}"; SHARD="$((T % 3))/3"
SNAP=$("$PLAN_PY" -c "import sys; sys.path.insert(0,'$PLAN_REPO/research/mars3_20260916/code'); import m3common as m; print(m.snap('${MODELS[$((T / 3))]}'))")
echo "[g2] task $T family $FAM shard $SHARD snapshot $SNAP"; plan_receipt "$G2/receipt_${FAM}_$((T % 3))"
for Q in coverage support; do
  "$PLAN_PY" "$MCCODE/mc_judge_precision.py" run --tasks "$G2/tasks.jsonl" --out "$G2/judgments/${FAM}_${Q}_shard$((T % 3)).jsonl" --model-snapshot "$SNAP" --question "$Q" --shard "$SHARD" --batch-size 8
done
plan_done "$G2/receipt_${FAM}_$((T % 3))"
