#!/bin/bash
# M-B.3 -- LLM-AggreFact evaluation of every verifier that has dev and test score files. CPU. Depends on the
# mb_score arrays. Registered in research/mars_metric_20260923/PREREG.md.
#
#   sbatch --dependency=afterok:<factcg array>:<marsc array> research/mars_metric_20260923/slurm/mb_eval.sbatch
#SBATCH --job-name=mars-mb-eval
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/results/mars_metric/logs/%x-%A.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
MM="${MARS_METRIC_OUT:-/home/user/results/mars_metric}"; CODE="$PLAN_REPO/research/mars_metric_20260923/code"
mkdir -p "$MM/mb/eval"; plan_receipt "$MM/mb/eval"
for V in factcg marsc; do
  D="$MM/mb/scores/scores_${V}_dev*.jsonl"; T="$MM/mb/scores/scores_${V}_test*.jsonl"
  ls $D >/dev/null 2>&1 && ls $T >/dev/null 2>&1 || { echo "[mb-eval] $V: score files missing, skipped"; continue; }
  "$PLAN_PY" "$CODE/mb_eval.py" --dev "$D" --test "$T" --verifier "$V" --n-boot 2000 --seed 20260918 --out "$MM/mb/eval/mb_eval_${V}.json"
done
(cd "$MM/mb" && find eval scores -type f -name "*.json*" | sort | xargs sha256sum > eval/SHA256SUMS); plan_done "$MM/mb/eval"
echo "[mb-eval] done"
