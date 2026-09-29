#!/bin/bash
# H-J11 (gap fill, 2026-09-27) -- resident weights of windowed RoBERTa-large-MNLI and AlignScore, which H-J5
# recorded as None because its probe looked for the wrong attribute names (fixed in hj5_deploy_cost.py). Same
# inventory, seed and batch as H-J5; written to a separate directory so the H-J5 receipts stay as they were.
#SBATCH --job-name=mars-hj11-deploy
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%j.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"
JCODE="$PLAN_REPO/research/marsc_ieee_20260922/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
OUT="$MCS/hj11/deploy"; mkdir -p "$OUT"; plan_receipt "$OUT"
INV="$M3/m32/units_gemma+ent_validation.jsonl"
MARSC_CKPT="$MC/e1/mc-A-lam1-crossenc-inv_seed20260906/model"
FACTCG=$(snap "yaxili96/FactCG-DeBERTa-v3-Large") || true
"$PLAN_PY" "$JCODE/hj5_deploy_cost.py" --backend hf --systems nli alignscore \
  --inventory "$INV" --marsc-ckpt "$MARSC_CKPT" --factcg-snapshot "$FACTCG" --out "$OUT" --batch 64
plan_done "$OUT"; echo "[hj11] done -> $OUT"
