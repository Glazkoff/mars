#!/bin/bash
# MARS-C G2 analysis: true-omission rate per emitter, judge validity on human-known units, family agreement, paired bootstrap.
#SBATCH --job-name=mc-g2-an
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"; G2="$MC/g2"
"$PLAN_PY" "$MCCODE/mc_judge_precision.py" analyze --tasks "$G2/tasks.jsonl" --families "gemma=$G2/judgments/gemma_*_shard*.jsonl" "qwen=$G2/judgments/qwen_*_shard*.jsonl" --primary "humanfact+div1" --primary-family gemma --n-boot 300 --out "$G2/analysis"
