#!/bin/bash
# M-F -- equivalence check: the released mars/ package vs the ma_score.py research-script numbers, on 200
# random pairs, tolerance 1e-3. Registered in research/mars_metric_20260923/PREREG.md.
#
#   sbatch research/mars_metric_20260923/slurm/mf_equiv.sbatch
#SBATCH --job-name=mars-mf-equiv
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user/results/mars_metric/logs/%x-%A.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"
MM="${MARS_METRIC_OUT:-/home/user/results/mars_metric}"; CODE="$PLAN_REPO/research/mars_metric_20260923/code"
mkdir -p "$MM/mf"; plan_receipt "$MM/mf/equiv"
SNAP=$(snap "yaxili96/FactCG-DeBERTa-v3-Large") || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PLAN_PY" "$CODE/mf_equiv.py" \
  --pairs "$MM/ma/tasks/pairs_ma.jsonl" \
  --props "$MM/ma/decomp/props_shard*.jsonl" \
  --score-p "$MM/ma/scores/scores_marsP_factcg.jsonl" \
  --score-r "$MM/ma/scores/scores_marsR_marsc.jsonl" \
  --factcg-snapshot "$SNAP" \
  --marsc-ckpts "$MC/e1/mc-A-lam1-crossenc-inv_seed*/model" \
  --n 200 --seed 20260924 --tol 1e-3 \
  --out "$MM/mf/equiv/mf_equiv.json"
plan_done "$MM/mf/equiv"
echo "[mf-equiv] done"
