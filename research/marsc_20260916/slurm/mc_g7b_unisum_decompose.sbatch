#!/bin/bash
# MARS-C G7b -- Gemma-4-31B sentence decomposition of the UniSumEval documents (the frozen inventory recipe), two shards.
#SBATCH --job-name=mc-g7b
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/user/results/marsc/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; G7="$MC/g7_unisum"; T="${SLURM_ARRAY_TASK_ID:-0}"
"$PLAN_PY" "$M3CODE/m32_decompose.py" --tasks "$G7/decompose_tasks_unisum.jsonl" --out "$G7/props/props_unisum_shard$T.jsonl" --shard "$T/2" --batch-size 12
