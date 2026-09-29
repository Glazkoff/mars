#!/bin/bash
# MARS-C R04b -- the SAME Gemma-4-31B-it coverage judge the paper's judge-label pipeline used, run over the
# identical human-ACU cells of the E0 TRAIN and CALIB roles. 8 shards, resumable by task_id.
#SBATCH --job-name=mc-r04b-judge
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/home/user/results/marsc/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"
SNAP="$(snap google/gemma-4-31B-it)"; [ -n "$SNAP" ] || { echo "[r04b] no cached gemma snapshot"; exit 3; }
i="${SLURM_ARRAY_TASK_ID:-0}"; OUT="$MC/r04_matched"; mkdir -p "$OUT/judge"; plan_receipt "$OUT/judge/receipt_$i"
"$PLAN_PY" scripts/plan2026/b1_judge_labels.py --tasks "$OUT/judge_tasks_acu_trainrole.jsonl" \
  --out "$OUT/judge/judge_shard${i}.jsonl" --model-snapshot "$SNAP" --shard "$i/8" --splits train,calib --batch-size 8
plan_done "$OUT/judge/receipt_$i"
