#!/bin/bash
# M-C -- score LLM-AggreFact dev and test with each unified-verifier seed through mb_score.py (FactCG contract,
# whole claim and decomposed). Array task t: seed = t / 2, split = dev if t even else test.
#
#   sbatch --dependency=afterok:<mc_train array> research/mars_metric_20260923/slurm/mc_score.sbatch
#SBATCH --job-name=mars-mc-score
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/user/results/mars_metric/logs/%x-%A_%a.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
MM="${MARS_METRIC_OUT:-/home/user/results/mars_metric}"; CODE="$PLAN_REPO/research/mars_metric_20260923/code"
SEEDS=(20260906 20260907 20260908); T="${SLURM_ARRAY_TASK_ID:-0}"; S="${SEEDS[$((T / 2))]}"; SPLIT=$([ $((T % 2)) = 0 ] && echo dev || echo test)
AGG=/home/user/hf_cache/hub/datasets--lytang--LLM-AggreFact/snapshots/981dfd0bd8e58e7238a9ab92b2e6ea44bce918e4/data
MODEL="$MM/mc/seed$S/model"; [ -f "$MODEL/config.json" ] || { echo "[mc-score] MISSING $MODEL"; exit 4; }
for t in 0 1; do [ -s "$MM/mb/decomp/props_shard$t.jsonl" ] || { echo "[mc-score] MISSING props shard $t"; exit 4; }; done
mkdir -p "$MM/mc/scores"; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[mc-score] seed $S split $SPLIT"
"$PLAN_PY" "$CODE/mb_score.py" --aggrefact-dir "$AGG" --props "$MM/mb/decomp/props_shard*.jsonl" --split "$SPLIT" --verifier factcg --snapshot "$MODEL" --batch 32 \
  --out "$MM/mc/scores/scores_seed${S}_${SPLIT}.jsonl" 2>&1 | grep -E "^\[mb\]|^\[b10\]|Error|Traceback" | grep -v "pairs (" || true
[ -s "$MM/mc/scores/scores_seed${S}_${SPLIT}.jsonl" ] || { echo "[mc-score] no output"; exit 5; }
echo "[mc-score] done"
