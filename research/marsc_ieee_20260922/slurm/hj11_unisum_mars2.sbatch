#!/bin/bash
# H-J11 (gap fill, 2026-09-27) -- MARS-2 (direct_enabled, the released direct-scoring arm) on the UniSumEval pairs,
# the same three seeds and the same dump call b2_score.sbatch used for the RoSE TEST rows of Table 1.
#SBATCH --job-name=mars-hj11-unisum-m2
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A_%a.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
SEEDS=(20260906 20260907 20260908); SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
CKPT="/home/user/models/plan2026/b2/direct_enabled_seed${SEED}/epoch1"; [ -d "$CKPT" ] || { echo "no $CKPT"; exit 3; }
OUT="$MCS/hj11/unisum_mars2/direct_enabled_seed${SEED}"; mkdir -p "$OUT"; plan_receipt "$OUT"
export MARS2_DEVICE=cuda MARS2_SPACY_MODEL=en_core_web_sm
"$PLAN_PY" -m mars_v2.dump --model "$CKPT" --pairs "$MC/g7_unisum/acu_units_unisum.jsonl" \
  --out "$OUT/units_unisum.jsonl" --id-key pair_id --source-key source --candidate-key candidate --units entity+proposition
sha256sum "$OUT"/*.jsonl > "$OUT/SHA256SUMS"; plan_done "$OUT"
