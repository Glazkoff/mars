#!/bin/bash
# MARS-C B8 -- the vLLM pilot. Registered: research/marsc_strengthen_20260918/PREREG.md, Wave B / B8. No bar;
# the brief's instruction is explicit -- report the outcome either way, and "vLLM does not load this
# architecture" is a campaign-shaping result.
#
# State of knowledge before this job: vLLM 0.29.0 is installed at /home/user/.venv-vllm (torch 2.13.0+cu130,
# transformers 5.17.0) and has NEVER run here -- every LLM log in the project records backend=hf and the Slurm
# node comment records "vLLM off". Read-only checks done on the login node before writing this job:
#   * Qwen/Qwen3.8-27B config.json declares architectures=['Qwen3_5ForConditionalGeneration'], model_type qwen3_5;
#   * vLLM 0.29.0's registry lists 376 architectures INCLUDING Qwen3_5ForConditionalGeneration and
#     Gemma4ForConditionalGeneration.
# Registry membership is NOT a load. This job is the load.
#
# Identification: both backends receive the SAME fully rendered prompt strings, produced once by the HF chat
# template with enable_thinking=False, from the frozen m32_decompose PROMPT and task contract (imported, not
# retyped). vLLM is driven through LLM.generate(prompts=...) and never LLM.chat(...), so no template
# difference can enter the comparison. The rendered prompt is asserted to end in a CLOSED empty think block
# before a single token is generated (prereg standing rule 7); blank-output rate is recorded for both arms.
#
# One debugging round, as instructed, and no more: attempt 1 lets vLLM use its own compilation path (this
# requires clearing the repo-wide TORCHDYNAMO_DISABLE=1 and expandable_segments, both of which are defensive
# defaults for HF training jobs, not for a vLLM engine); attempt 2, only if attempt 1 did not load, is
# --enforce-eager. Whatever happens, the HF same-model arm still runs and the verdict is still written.
#
# Documents are drawn from the RoSE VALIDATION split, so no sealed-TEST re-read and no MC_TEST_REREAD gate.
# The generations are a throughput artifact and enter no inventory, training set, calibration or selection.
#SBATCH --job-name=mars-b8-vllm-pilot
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:30:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
OUT="$MS/b6b8/b8"; mkdir -p "$OUT"; plan_receipt "$OUT"
VLLM_PY="${B8_VLLM_PY:-/home/user/.venv-vllm/bin/python}"
TASKS="$M3/m32/decompose_tasks.jsonl"
QWEN=$(snap "Qwen/Qwen3.8-27B")
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

[ -x "$VLLM_PY" ] || { echo "[b8] MISSING vllm interpreter $VLLM_PY" >&2; exit 4; }
[ -f "$TASKS" ] || { echo "[b8] MISSING $TASKS" >&2; exit 4; }
if [ -z "$QWEN" ] || [ ! -f "$QWEN/config.json" ]; then echo "[b8] MISSING Qwen snapshot" >&2; exit 4; fi
# trap 7: a metadata-only stub still resolves to a path and fails at load
QSZ=$(du -sm --dereference "$QWEN" 2>/dev/null | awk '{print $1}')
echo "[b8] qwen snapshot $QWEN (${QSZ} MiB), tasks $(wc -l < "$TASKS") rows"
[ "${QSZ:-0}" -ge 20000 ] || { echo "[b8] Qwen snapshot looks like a metadata-only stub (${QSZ} MiB)" >&2; exit 4; }
"$VLLM_PY" -c "import vllm, torch, transformers; print('[b8] vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

loaded () {  # loaded <json> -> "1" when the record says the backend came up
  "$PLAN_PY" - "$1" <<'PY' 2>/dev/null || echo 0
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print("1" if (d.get("result") or {}).get("loaded") else "0")
except Exception:
    print("0")
PY
}

N_DOCS_V="${B8_N_DOCS:-50}"; N_DOCS_H="${B8_N_DOCS_HF:-12}"

echo "[b8] ===== attempt 1: vLLM, own compilation path ====="
set +e
env -u TORCHDYNAMO_DISABLE -u PYTORCH_CUDA_ALLOC_CONF \
  "$VLLM_PY" "$SCODE/b8_vllm_pilot.py" --backend vllm --name vllm \
  --tasks "$TASKS" --splits validation --n-docs "$N_DOCS_V" \
  --model-id "Qwen/Qwen3.8-27B" --model-snapshot "$QWEN" \
  --max-new-tokens 160 --max-model-len 1024 \
  --gpu-memory-utilization "${B8_GPU_UTIL:-0.85}" --out "$OUT"
echo "[b8] attempt 1 exit=$?"
set -e

VJSON="$OUT/b8_vllm.json"
if [ "$(loaded "$VJSON")" != "1" ]; then
  echo "[b8] ===== attempt 2 (the one debugging round): vLLM --enforce-eager ====="
  set +e
  env -u TORCHDYNAMO_DISABLE -u PYTORCH_CUDA_ALLOC_CONF \
    "$VLLM_PY" "$SCODE/b8_vllm_pilot.py" --backend vllm --name vllm_eager --enforce-eager \
    --tasks "$TASKS" --splits validation --n-docs "$N_DOCS_V" \
    --model-id "Qwen/Qwen3.8-27B" --model-snapshot "$QWEN" \
    --max-new-tokens 160 --max-model-len 1024 \
    --gpu-memory-utilization "${B8_GPU_UTIL:-0.85}" --out "$OUT"
  echo "[b8] attempt 2 exit=$?"
  set -e
  if [ "$(loaded "$OUT/b8_vllm_eager.json")" = "1" ]; then VJSON="$OUT/b8_vllm_eager.json"; fi
fi

echo "[b8] ===== HF same-model baseline (the only controlled denominator) ====="
set +e
"$PLAN_PY" "$SCODE/b8_vllm_pilot.py" --backend hf --name hf \
  --tasks "$TASKS" --splits validation --n-docs "$N_DOCS_H" \
  --model-id "Qwen/Qwen3.8-27B" --model-snapshot "$QWEN" \
  --max-new-tokens 160 --max-model-len 1024 --batch-size "${B8_HF_BATCH:-12}" --out "$OUT"
echo "[b8] hf arm exit=$?"
set -e

"$PLAN_PY" "$SCODE/b8_compare.py" --vllm "$VJSON" --hf "$OUT/b8_hf.json" --out "$OUT"
(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS)
plan_done "$OUT"
echo "[b8] done"
