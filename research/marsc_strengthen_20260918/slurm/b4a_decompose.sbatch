#!/bin/bash
# MARS-C strengthening B4a -- the LITERAL UNSEEN THIRD EXTRACTOR decomposes the held-out source sentences.
#
# Registered in research/marsc_strengthen_20260918/PREREG.md, Wave B / B4. Qwen/Qwen3.8-27B replaces
# Gemma-4-31B-it as the atomic-proposition extractor and NOTHING else changes: the prompt literal and the
# parser are IMPORTED from the declared interface (m32_decompose.py) rather than retyped, the sentence task
# file is the frozen m32 one filtered to the held-out split, and the decoding contract (chat template with
# enable_thinking=False, left padding, truncation at 1024, greedy, 160 new tokens, first 12 propositions) is
# carried over unchanged. None of this extractor's units, scores, labels or output statistics may inform
# training, mining, calibration or model selection.
#
# The freeze manifest -- repo id, resolved snapshot revision, sha256 of the prompt literal, sha256 of the task
# file, every decoding parameter -- is written by the driver BEFORE the model is loaded, i.e. before a single
# unit of the unseen extractor exists.
#
# TRAP GUARDED: Qwen3's chat template opens a <think> block unless enable_thinking=False reaches it; that
# emptied 88 % of outputs in an earlier campaign and produced a wrong published result. The driver refuses to
# generate unless the RENDERED prefix ends in a closed empty think block, and aborts non-zero if the blank
# decode rate exceeds 10 % or the empty-parse rate exceeds 30 % (the development extractor's own empty-parse
# rate on these sentences is 10.43 %, so a 10 % parse guard would fail the healthy control -- see the driver
# header for the declared deviation).
#SBATCH --job-name=mars-b4a-qwen-decompose
# PARTITION NOTE: GPU work runs on `infer`, not `train`. Both partitions serve the same eight H200s on
# the same node, `infer` carries PriorityTier=100 against train's 1, and every existing MARS-C GPU job
# (mc_e1, mc_g1_test, mc_g7b) was submitted there. At submission `train` held eighteen pending GPU jobs
# and a nine-hour start estimate. Nothing is cancelled, requeued or preempted by this choice.
#SBATCH --partition=infer
#SBATCH --nodelist=Euler
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A_%a.out
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[b4a] REFUSED: MC_TEST_REREAD=yes not set (sealed TEST split)"; exit 3; }
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
OUT="${B4_OUT:-/home/user/results/marsc_strengthen/b4}"; T="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$OUT/props" "$OUT/freeze"
if [ "$T" = "0" ]; then plan_receipt "$OUT"; fi

SNAP="/home/user/hf_cache/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TASKS="$M3/m32/decompose_tasks.jsonl"
for f in "$TASKS" "$SNAP/config.json" "$SNAP/model.safetensors.index.json"; do
  [ -e "$f" ] || { echo "[b4a] MISSING $f"; exit 4; }
done
SZ=$(du -sLm "$SNAP" | cut -f1)
[ "$SZ" -ge 10000 ] || { echo "[b4a] REFUSED: snapshot is ${SZ} MB -- a metadata-only stub, not weights"; exit 5; }
echo "[b4a] snapshot ${SZ} MB at $SNAP"

"$PLAN_PY" "$SCODE/b4_decompose.py" \
  --tasks "$TASKS" --splits test \
  --out "$OUT/props/props_qwen_test_shard$T.jsonl" \
  --freeze-out "$OUT/freeze/freeze_shard$T.json" \
  --model-id "Qwen/Qwen3.8-27B" --model-snapshot "$SNAP" \
  --shard "$T/2" --batch-size 12 --max-new-tokens 160
echo "[b4a] shard $T done"
