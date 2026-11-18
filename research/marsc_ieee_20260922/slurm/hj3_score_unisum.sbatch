#!/bin/bash
# H-J3b -- score the frozen UniSumEval Gemma+ent inventory with one of the three 2026 comparators, in all
# three candidate modes (real, shuffled, empty), with the repaired scorers of 2026-09-21. Registered in
# research/marsc_ieee_20260922/PREREG.md BEFORE submission. One submission per system:
#
#   sbatch --export=ALL,HJ3_SYSTEM=judge      research/marsc_ieee_20260922/slurm/hj3_score_unisum.sbatch
#   sbatch --export=ALL,HJ3_SYSTEM=factcg     research/marsc_ieee_20260922/slurm/hj3_score_unisum.sbatch
#   sbatch --export=ALL,HJ3_SYSTEM=minicheck7b research/marsc_ieee_20260922/slurm/hj3_score_unisum.sbatch
#
# Mirrors b9_judge.sbatch / b10_verifier_swap.sbatch with the inventory path pointed at the external pool
# and the outputs kept under hj3/, so the RoSE score directories the paper's rows came from are untouched.
#SBATCH --job-name=mars-hj3-score
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=07:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
SYSTEM="${HJ3_SYSTEM:?set HJ3_SYSTEM=judge|factcg|minicheck7b}"
MODES="${HJ3_MODES:-real shuffled empty}"
# H-J12 reuses this job on the distilled UniSumEval inventory through HJ3_INV / HJ3_INVFILE / HJ3_OUT; the defaults
# are the H-J3b run of record.
INV="${HJ3_INV:-gemma+ent}"; INVFILE="${HJ3_INVFILE:-$MC/g7_unisum/units_gemma+ent_unisum.jsonl}"
[ -f "$INVFILE" ] || { echo "[hj3] MISSING $INVFILE"; exit 4; }
OUT="${HJ3_OUT:-$MCS/hj3/scores_unisum/$SYSTEM}"; mkdir -p "$OUT/scores" "$MCS/logs"; plan_receipt "$OUT"
VLLM_PY="${HJ3_VLLM_PY:-/home/user/.venv-vllm/bin/python}"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="/home/user/.venv-vllm/bin:$CUDA_HOME/bin:$PATH"
echo "[hj3] system=$SYSTEM modes='$MODES' inventory=$INVFILE out=$OUT"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

case "$SYSTEM" in
  judge)
    MODEL_ID="google/gemma-4-31B-it"; SNAP=$(snap "$MODEL_ID") || true
    [ -n "$SNAP" ] && [ -f "$SNAP/config.json" ] || { echo "[hj3] MISSING snapshot for $MODEL_ID"; exit 4; }
    command -v ninja >/dev/null || { echo "[hj3] ninja not on PATH"; exit 4; }
    for MODE in $MODES; do
      echo "===== [hj3] judge candidate-mode=$MODE"
      env -u TORCHDYNAMO_DISABLE -u PYTORCH_CUDA_ALLOC_CONF "$VLLM_PY" "$SCODE/b9_judge_units.py" \
        --units "$INV=$INVFILE" --model-id "$MODEL_ID" --model-snapshot "$SNAP" --tag gemma31 \
        --candidate-mode "$MODE" --shuffle-seed 20260917 --scores-out "$OUT/scores" --score-mode exact \
        --max-model-len "${HJ3_JUDGE_LEN:-8192}" --gpu-memory-utilization "${HJ3_GPU_UTIL:-0.90}"
    done ;;
  factcg)
    MODEL_ID="yaxili96/FactCG-DeBERTa-v3-Large"; SNAP=$(snap "$MODEL_ID") || true
    [ -n "$SNAP" ] && [ -f "$SNAP/config.json" ] || { echo "[hj3] MISSING snapshot for $MODEL_ID"; exit 4; }
    AGG="${HJ3_AGGREFACT_JSONL:-$MCS/b10/aggrefact_sample.jsonl}"; [ -f "$AGG" ] || { echo "[hj3] MISSING $AGG"; exit 4; }
    FIRST=1
    for MODE in $MODES; do
      echo "===== [hj3] factcg candidate-mode=$MODE"
      VAL=""; [ "$FIRST" = 1 ] && VAL="--validate-aggrefact ${HJ3_VALIDATE:-500}"
      env -u TORCHDYNAMO_DISABLE -u PYTORCH_CUDA_ALLOC_CONF "$PLAN_PY" "$SCODE/b10_verifier_swap.py" \
        --units "$INV=$INVFILE" --system factcg --snapshot "$SNAP" --candidate-mode "$MODE" \
        --shuffle-seed 20260917 --scores-out "$OUT/scores" --aggrefact-jsonl "$AGG" $VAL
      FIRST=0
    done ;;
  minicheck7b)
    MINICHECK_DIR="${HJ3_MINICHECK_DIR:-/home/user/models/minicheck7b_native}"
    [ -f "$MINICHECK_DIR/config.json" ] || { echo "[hj3] MISSING sanitized snapshot $MINICHECK_DIR"; exit 4; }
    AGG="${HJ3_AGGREFACT_JSONL:-$MCS/b10/aggrefact_sample.jsonl}"; [ -f "$AGG" ] || { echo "[hj3] MISSING $AGG"; exit 4; }
    command -v ninja >/dev/null || { echo "[hj3] ninja not on PATH"; exit 4; }
    export VLLM_ENABLE_V1_MULTIPROCESSING=0
    FIRST=1
    for MODE in $MODES; do
      echo "===== [hj3] minicheck7b candidate-mode=$MODE"
      VAL=""; [ "$FIRST" = 1 ] && VAL="--validate-aggrefact ${HJ3_VALIDATE:-500}"
      env -u TORCHDYNAMO_DISABLE -u PYTORCH_CUDA_ALLOC_CONF "$VLLM_PY" "$SCODE/b10_verifier_swap.py" \
        --units "$INV=$INVFILE" --system minicheck7b --snapshot "$MINICHECK_DIR" --candidate-mode "$MODE" \
        --shuffle-seed 20260917 --scores-out "$OUT/scores" --max-model-len "${HJ3_MC_LEN:-8192}" \
        --gpu-memory-utilization "${HJ3_GPU_UTIL:-0.90}" --trust-remote-code --aggrefact-jsonl "$AGG" $VAL
      FIRST=0
    done ;;
  *) echo "[hj3] unknown HJ3_SYSTEM=$SYSTEM"; exit 2 ;;
esac

N=$(ls "$OUT/scores"/${INV}_*.jsonl 2>/dev/null | wc -l); NM=$(echo $MODES | wc -w)
[ "$N" -ge "$NM" ] || { echo "[hj3] expected $NM score files (one per mode), found $N"; exit 5; }
(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS)
plan_done "$OUT"
echo "[hj3] done -> $OUT/scores ($N files)"
