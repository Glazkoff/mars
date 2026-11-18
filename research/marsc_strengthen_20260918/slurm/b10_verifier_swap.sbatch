#!/bin/bash
# MARS-C B10 -- the 2025/26 LLM-AggreFact leaders as substitutable verifiers.
# MARS_PUBLICATION_PLAN_2026-09-20.md Sec. 3.2, row 2.
#
# FactCG (HF cross-encoder, .venv-mars) on sealed TEST, all three candidate modes:
#   sbatch --export=ALL,B10_SYSTEM=factcg,B10_SPLIT=test,MC_TEST_REREAD=yes,B10_MODES="real shuffled empty" \
#          research/marsc_strengthen_20260918/slurm/b10_verifier_swap.sbatch
#
# Bespoke-MiniCheck-7B (vLLM, .venv-vllm):
#   sbatch --export=ALL,B10_SYSTEM=minicheck7b,B10_SPLIT=test,MC_TEST_REREAD=yes,B10_MODES="real shuffled empty" \
#          research/marsc_strengthen_20260918/slurm/b10_verifier_swap.sbatch
#
#SBATCH --job-name=mars-b10-swap
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh

SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
SYSTEM="${B10_SYSTEM:?set B10_SYSTEM=factcg|minicheck7b}"
SPLIT="${B10_SPLIT:-validation}"
MODES="${B10_MODES:-real}"
LIMIT="${B10_LIMIT:-0}"
NVAL="${B10_VALIDATE:-2000}"

if [ "$SPLIT" = "test" ]; then
  [ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[b10] REFUSED: sealed TEST split needs MC_TEST_REREAD=yes"; exit 3; }
fi

INV="${B10_INV:-distilled}"
INVFILE="$M3/m32/units_${INV}_${SPLIT}.jsonl"
[ -f "$INVFILE" ] || { echo "[b10] MISSING $INVFILE"; exit 4; }

case "$SYSTEM" in
  factcg)
    MODEL_ID="yaxili96/FactCG-DeBERTa-v3-Large"; PY="$PLAN_PY"
    # the SAME dumped benchmark sample MiniCheck-7B is gated on, so the two gates are comparable
    EXTRA="--aggrefact-jsonl ${B10_AGGREFACT_JSONL:-/home/user/results/marsc_strengthen/b10/aggrefact_sample.jsonl}" ;;
  minicheck7b)
    MODEL_ID="bespokelabs/Bespoke-MiniCheck-7B"
    # Served from the sanitized snapshot built by b10_prepare_minicheck.py: symlinked weights with `auto_map`
    # stripped, so vLLM uses its native InternLM2 implementation and the serialized fast tokenizer. The
    # checkpoint's bundled code is unusable here -- its forward() lacks `intermediate_tensors` (job 5660) and
    # its tokenizer falls back to byte level. See that script's docstring.
    MINICHECK_DIR="${B10_MINICHECK_DIR:-/home/user/models/minicheck7b_native}"
    PY="${B10_VLLM_PY:-/home/user/.venv-vllm/bin/python}"
    # --trust-remote-code is still needed: the sanitized config keeps its AutoConfig entry so transformers 5,
    # which has no native `internlm2` config, can parse it at all. The model entries are gone, so the weights
    # resolve through `architectures` to vLLM's own InternLM2ForCausalLM.
    # The vLLM venv has no `datasets`, so the benchmark gate reads a sample dumped once from the mars venv.
    AGG_JSONL="${B10_AGGREFACT_JSONL:-/home/user/results/marsc_strengthen/b10/aggrefact_sample.jsonl}"
    [ -f "$AGG_JSONL" ] || { echo "[b10] MISSING benchmark sample $AGG_JSONL (run --dump-aggrefact first)"; exit 4; }
    EXTRA="--max-model-len ${B10_MAX_LEN:-4096} --gpu-memory-utilization ${B10_GPU_UTIL:-0.90} --trust-remote-code --aggrefact-jsonl $AGG_JSONL"
    export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
    # The engine must run IN THIS PROCESS: b10_verifier_swap.py restores the missing default on vLLM's own
    # InternLM2ForCausalLM.forward(), and that patch would not survive a fork into a separate EngineCore.
    export VLLM_ENABLE_V1_MULTIPROCESSING=0
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export PATH="/home/user/.venv-vllm/bin:$CUDA_HOME/bin:$PATH"
    command -v ninja >/dev/null || { echo "[b10] ninja not on PATH"; exit 4; } ;;
  *) echo "[b10] unknown B10_SYSTEM=$SYSTEM"; exit 2 ;;
esac

[ -x "$PY" ] || { echo "[b10] MISSING interpreter $PY"; exit 4; }
if [ -n "${MINICHECK_DIR:-}" ]; then
  SNAP="$MINICHECK_DIR"
  [ -f "$SNAP/config.json" ] || { echo "[b10] MISSING sanitized snapshot $SNAP -- run b10_prepare_minicheck.py"; exit 4; }
else
  SNAP=$(snap "$MODEL_ID") || true
  if [ -z "$SNAP" ] || [ ! -f "$SNAP/config.json" ]; then echo "[b10] MISSING snapshot for $MODEL_ID"; exit 4; fi
fi

OUT="$MCS/b10/$SPLIT"; mkdir -p "$OUT/scores" "$MCS/logs"; plan_receipt "$OUT"
echo "[b10] system=$SYSTEM split=$SPLIT inv=$INV modes='$MODES' model=$MODEL_ID"
echo "[b10] snapshot=$SNAP interpreter=$PY"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

FIRST=1
for MODE in $MODES; do
  echo "===== [b10] $SYSTEM candidate-mode=$MODE"
  # The benchmark re-measurement is a property of the model, not of the mode: run it once.
  VAL_ARG=""
  [ "$FIRST" = "1" ] && [ "$NVAL" -gt 0 ] && VAL_ARG="--validate-aggrefact $NVAL"
  LIMIT_ARG=""
  [ "$LIMIT" -gt 0 ] && LIMIT_ARG="--limit $LIMIT"
  env -u TORCHDYNAMO_DISABLE -u PYTORCH_CUDA_ALLOC_CONF \
    "$PY" "$SCODE/b10_verifier_swap.py" \
      --units "$INV=$INVFILE" \
      --system "$SYSTEM" --snapshot "$SNAP" \
      --candidate-mode "$MODE" --shuffle-seed 20260917 \
      --scores-out "$OUT/scores" \
      $EXTRA $VAL_ARG $LIMIT_ARG
  FIRST=0
done

plan_done "$OUT"
echo "[b10] done -> $OUT"
