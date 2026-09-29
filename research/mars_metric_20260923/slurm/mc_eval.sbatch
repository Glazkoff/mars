#!/bin/bash
# M-C -- per-seed LLM-AggreFact evaluation (mb_eval.py) and the registered bars over the three seeds. CPU.
#
#   sbatch --dependency=afterok:<mc_score array> research/mars_metric_20260923/slurm/mc_eval.sbatch
#SBATCH --job-name=mars-mc-eval
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/results/mars_metric/logs/%x-%A.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
MM="${MARS_METRIC_OUT:-/home/user/results/mars_metric}"; CODE="$PLAN_REPO/research/mars_metric_20260923/code"
mkdir -p "$MM/mc/eval"; plan_receipt "$MM/mc/eval"
[ -f "$MM/mb/eval/mb_eval_factcg.json" ] || { echo "[mc-eval] MISSING the M-B A1 reference"; exit 4; }
R=(); E=()
for S in 20260906 20260907 20260908; do
  "$PLAN_PY" "$CODE/mb_eval.py" --dev "$MM/mc/scores/scores_seed${S}_dev*.jsonl" --test "$MM/mc/scores/scores_seed${S}_test*.jsonl" --verifier "unified_seed$S" --n-boot 2000 --seed 20260918 --out "$MM/mc/eval/mb_eval_seed$S.json"
  R+=("$MM/mc/seed$S/result.json"); E+=("$MM/mc/eval/mb_eval_seed$S.json")
done
"$PLAN_PY" "$CODE/mc_bars.py" --results "${R[@]}" --mb-evals "${E[@]}" --a1 "$MM/mb/eval/mb_eval_factcg.json" --out "$MM/mc/eval/mc_bars.json"
(cd "$MM/mc" && find eval -type f -name "*.json" | sort | xargs sha256sum > eval/SHA256SUMS; for S in 20260906 20260907 20260908; do sha256sum "seed$S/result.json" >> eval/SHA256SUMS; done); plan_done "$MM/mc/eval"
echo "[mc-eval] done"
