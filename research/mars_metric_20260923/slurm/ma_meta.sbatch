#!/bin/bash
# M-A.3 -- meta-evaluation over every score file present. CPU. Depends on the ma_score array (afterany, so that
# one failed comparator does not block the report; the report names the columns it found).
#
#   sbatch --dependency=afterany:<ma_score array> research/mars_metric_20260923/slurm/ma_meta.sbatch
#SBATCH --job-name=mars-ma-meta
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/results/mars_metric/logs/%x-%A.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
MM="${MARS_METRIC_OUT:-/home/user/results/mars_metric}"; CODE="$PLAN_REPO/research/mars_metric_20260923/code"
mkdir -p "$MM/ma/meta"; plan_receipt "$MM/ma/meta"
ls "$MM/ma/scores"/scores_*.jsonl
"$PLAN_PY" "$CODE/ma_meta.py" --pairs "$MM/ma/tasks/pairs_ma.jsonl" --scores "$MM/ma/scores/scores_*.jsonl" --n-boot 2000 --seed 20260918 --out "$MM/ma/meta/meta_evaluation.json"
(cd "$MM/ma" && find meta scores -type f -name "*.json*" | sort | xargs sha256sum > meta/SHA256SUMS); plan_done "$MM/ma/meta"
echo "[ma-meta] done"
