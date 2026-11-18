#!/bin/bash
# MARS-C R04a -- build the judge tasks for the matched-unit label-source control, and count them. CPU only.
#SBATCH --job-name=mc-r04a-tasks
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
OUT="$MC/r04_matched"; mkdir -p "$OUT/judge"; plan_receipt "$OUT"
"$PLAN_PY" "$MCCODE/mc_r04_relabel.py" tasks --docs-dir "$MC/e0" --roles train,calib --out "$OUT/judge_tasks_acu_trainrole.jsonl"
wc -l "$OUT/judge_tasks_acu_trainrole.jsonl"
