#!/bin/bash
# MARS-C B9 -- the zero-shot LLM judge as an omission finder on the EMITTED endpoint.
# MARS_PUBLICATION_PLAN_2026-09-20.md Sec. 3.2, row 1. Scores the frozen inventory with a 31B instruct model
# under the same premise / claim / output contract as every other off-the-shelf verifier, so recall@10,
# omission precision and H-COND for the judge come out of the SAME evaluators as the published rows.
#
# Pilot (validation, 40 pairs, one GPU, sizes the full run):
#   sbatch --export=ALL,B9_SPLIT=validation,B9_LIMIT=40,B9_MODES=real \
#          research/marsc_strengthen_20260918/slurm/b9_judge.sbatch
#
# Full sealed-TEST run (ONE task -- see the sharding note below):
#   sbatch --export=ALL,B9_SPLIT=test,MC_TEST_REREAD=yes,B9_MODES="real shuffled empty",B9_MAX_LEN=4096 \
#          research/marsc_strengthen_20260918/slurm/b9_judge.sbatch
#
# SHARDING IS NOT WIRED END TO END. --shard writes parts/<...>.part<i>of<n>.jsonl, but b9b10_eval.sbatch
# globs the score directory and has NO merge stage, so sharded parts would enter mc_e2_eval as several
# disjoint emitters instead of one -- emptying its common-pair intersection and corrupting EVERY row in the
# table, the MARS-C arms of record included. Measured throughput is 220-350 units/s, i.e. ~15 min for the
# whole of sealed TEST on one GPU, so sharding buys nothing. The guard below refuses an array submission
# that has not set B9_NSHARD to match.
#
#SBATCH --job-name=mars-b9-judge
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A_%a.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh

SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
SPLIT="${B9_SPLIT:-validation}"
MODES="${B9_MODES:-real}"
LIMIT="${B9_LIMIT:-0}"
NSHARD="${B9_NSHARD:-1}"
TAG="${B9_TAG:-gemma31}"
MODEL_ID="${B9_MODEL_ID:-google/gemma-4-31B-it}"
SHARD_I="${SLURM_ARRAY_TASK_ID:-0}"
# An --array submission that forgot B9_NSHARD would make every task score the FULL inventory and race on the
# same unsuffixed output filename. And even a correct sharded run has no merge stage downstream (see above).
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ "${B9_NSHARD:-1}" -le 1 ]; then
  echo "[b9] REFUSED: --array submitted without B9_NSHARD>1; every task would score the full inventory and"
  echo "[b9]          overwrite the same file. Submit a single task, or set B9_NSHARD and add a merge stage."
  exit 3
fi
if [ "$SHARD_I" -ge "${B9_NSHARD:-1}" ]; then
  echo "[b9] REFUSED: shard index $SHARD_I >= B9_NSHARD=${B9_NSHARD:-1}"; exit 3
fi

# Sealed-TEST gate: the held-out split is only re-read on an explicit, declared opt-in.
if [ "$SPLIT" = "test" ]; then
  [ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[b9] REFUSED: sealed TEST split needs MC_TEST_REREAD=yes"; exit 3; }
fi

# Sec. 3.2 names the gemma+ent inventory; the distilled one is the second emitter of record in the main
# table, so both are scorable and the arm is named after whichever is used.
INV="${B9_INV:-distilled}"
INVFILE="$M3/m32/units_${INV}_${SPLIT}.jsonl"
[ -f "$INVFILE" ] || { echo "[b9] MISSING $INVFILE"; exit 4; }

VLLM_PY="${B9_VLLM_PY:-/home/user/.venv-vllm/bin/python}"
[ -x "$VLLM_PY" ] || { echo "[b9] MISSING vllm interpreter $VLLM_PY"; exit 4; }

SNAP=$(snap "$MODEL_ID") || true
if [ -z "$SNAP" ] || [ ! -f "$SNAP/config.json" ]; then echo "[b9] MISSING snapshot for $MODEL_ID"; exit 4; fi

# B8b's environment recipe: FlashInfer's JIT needs ninja and nvcc on PATH, and this project invokes venvs by
# interpreter path without activating them, so neither is there by default.
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="/home/user/.venv-vllm/bin:$CUDA_HOME/bin:$PATH"
command -v ninja >/dev/null || { echo "[b9] ninja not on PATH"; exit 4; }

OUT="$MCS/b9/$SPLIT"; mkdir -p "$OUT/scores" "$MCS/logs"; plan_receipt "$OUT"
echo "[b9] split=$SPLIT inv=$INV modes='$MODES' shard=$SHARD_I/$NSHARD limit=$LIMIT model=$MODEL_ID"
echo "[b9] snapshot=$SNAP"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

for MODE in $MODES; do
  echo "===== [b9] candidate-mode=$MODE shard=$SHARD_I/$NSHARD"
  SHARD_ARG=""
  [ "$NSHARD" -gt 1 ] && SHARD_ARG="--shard $SHARD_I/$NSHARD"
  LIMIT_ARG=""
  [ "$LIMIT" -gt 0 ] && LIMIT_ARG="--limit $LIMIT"
  env -u TORCHDYNAMO_DISABLE -u PYTORCH_CUDA_ALLOC_CONF \
    "$VLLM_PY" "$SCODE/b9_judge_units.py" \
      --units "$INV=$INVFILE" \
      --model-id "$MODEL_ID" --model-snapshot "$SNAP" --tag "$TAG" \
      --candidate-mode "$MODE" --shuffle-seed 20260917 \
      --scores-out "$OUT/scores" \
      --max-model-len "${B9_MAX_LEN:-2048}" \
      --gpu-memory-utilization "${B9_GPU_UTIL:-0.90}" \
      $SHARD_ARG $LIMIT_ARG
done

plan_done "$OUT"
echo "[b9] done -> $OUT"
