#!/bin/bash
# M-A (iii) -- importance-tagger scores over the UniSumEval source inventory (the E2 taggers, frozen, three seeds),
# for the reference-free MARS-R reading. RoSE test already has them (r10_policy_test). One GPU, minutes.
#SBATCH --job-name=mars-ma-imp
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/results/mars_metric/logs/%x-%A.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MM="${MARS_METRIC_OUT:-/home/user/results/mars_metric}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
OUT="$MM/ma/importance"; mkdir -p "$OUT"; plan_receipt "$OUT"
"$PLAN_PY" "$MCCODE/mc_score_importance.py" --ckpts "$MC/e2imp_all/tagger_seed*" --units "gemma+ent=$MC/g7_unisum/units_gemma+ent_unisum.jsonl" --tag importance_all --scores-out "$OUT" 2>&1 | grep -v Warning | tail -n 5
ls "$OUT"; plan_done "$OUT"; echo "[ma-imp] done"
