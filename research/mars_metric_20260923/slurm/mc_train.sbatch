#!/bin/bash
# M-C -- train the unified verifier (one array task per seed) and evaluate its coverage axis. Submitted after the
# M-B A1 number exists (afterok on mb_eval), as PREREG.md requires. One GPU per seed.
#
#   sbatch --dependency=afterok:<mb_eval> research/mars_metric_20260923/slurm/mc_train.sbatch
#SBATCH --job-name=mars-mc-train
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/user/results/mars_metric/logs/%x-%A_%a.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"
MM="${MARS_METRIC_OUT:-/home/user/results/mars_metric}"; CODE="$PLAN_REPO/research/mars_metric_20260923/code"
SEEDS=(20260906 20260907 20260908); S="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
SNAP=$(snap "yaxili96/FactCG-DeBERTa-v3-Large") || true; [ -f "$SNAP/config.json" ] || { echo "[mc] MISSING FactCG"; exit 4; }
C2D=$(ls /home/user/data/mars_metric/minicheck/data/c2d-*.parquet); D2C=$(ls /home/user/data/mars_metric/minicheck/data/d2c-*.parquet)
for f in "$C2D" "$D2C" "$MC/e0/docs_train.jsonl" "$MC/e0/docs_validation.jsonl" "$G/b18/acu_units_validation.jsonl"; do [ -f "$f" ] || { echo "[mc] MISSING $f"; exit 4; }; done
OUT="$MM/mc/seed$S"; mkdir -p "$OUT"; plan_receipt "$OUT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[mc] seed $S init $SNAP"
"$PLAN_PY" "$CODE/mc_unified_train.py" --init "$SNAP" --synth "$C2D" "$D2C" --cells-docs "$MC/e0/docs_train.jsonl" \
  --eval-acu "$G/b18/acu_units_validation.jsonl" --eval-docs "$MC/e0/docs_validation.jsonl" \
  --deployed-ckpts "$MC/e1/mc-A-lam1-crossenc-inv_seed*/model" --steps 1500 --lr 1e-5 --batch 16 --seed "$S" --out "$OUT" \
  2>&1 | grep -E "^\[mc\]|^\[mb\] marsc|Error|Traceback|OutOfMemory"
[ -f "$OUT/model/config.json" ] && [ -f "$OUT/result.json" ] || { echo "[mc] seed $S produced no model"; exit 5; }
plan_done "$OUT"; echo "[mc] done -> $OUT"
