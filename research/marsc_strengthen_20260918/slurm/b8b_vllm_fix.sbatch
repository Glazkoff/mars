#!/bin/bash
# MARS-C B8b -- the corrected vLLM pilot, and the environment recipe every future vLLM job on this box needs.
# Registered: research/marsc_strengthen_20260918/PREREG.md, Wave B / B8. No bar.
#
# WHAT JOB 5346 ESTABLISHED, which is NOT what the block feared.
# vLLM 0.29.0 LOADS Qwen/Qwen3.8-27B (Qwen3_5ForConditionalGeneration) on Euler: weights in 5.74 s, GPU KV
# cache 165,376 tokens, 51 PIECEWISE + 51 FULL CUDA graph shapes captured. Both attempts then died at the
# LAST step of engine init, in FlashInfer's JIT build of the top-k/top-p sampling kernel:
#     FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
# Diagnosis (read-only checks on the login node): `ninja` IS installed, at
# /home/user/.venv-vllm/bin/ninja (1.13.2) -- but this project invokes venvs by INTERPRETER PATH and never
# activates them, so .venv-vllm/bin is not on PATH. `nvcc` is likewise off PATH (/usr/local/cuda/bin/nvcc) and
# CUDA_HOME is unset. The failure is environmental, not architectural.
#
# So this job is a single corrected attempt with a declared fallback, no installs:
#   attempt 1  PATH += .venv-vllm/bin and /usr/local/cuda/bin, CUDA_HOME set -- FlashInfer JIT can build, and
#              vLLM keeps its default (fastest) sampling path;
#   attempt 2  additionally VLLM_USE_FLASHINFER_SAMPLER=0 -- the PyTorch sampler, which needs no JIT at all.
#              Runs only if attempt 1 still did not load.
# Decoding is greedy, so the sampler choice cannot change a single generated token; it changes only whether
# a kernel has to be compiled.
#
# Depends on 5346 (afterany) so the HF same-model arm's record exists and the verdict is the controlled one.
#SBATCH --job-name=mars-b8b-vllm-fix
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
OUT="$MS/b6b8/b8"; mkdir -p "$OUT"
VLLM_PY="${B8_VLLM_PY:-/home/user/.venv-vllm/bin/python}"
TASKS="$M3/m32/decompose_tasks.jsonl"
QWEN=$(snap "Qwen/Qwen3.8-27B")
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="/home/user/.venv-vllm/bin:$CUDA_HOME/bin:$PATH"

[ -x "$VLLM_PY" ] || { echo "[b8b] MISSING vllm interpreter $VLLM_PY" >&2; exit 4; }
[ -f "$TASKS" ] || { echo "[b8b] MISSING $TASKS" >&2; exit 4; }
if [ -z "$QWEN" ] || [ ! -f "$QWEN/config.json" ]; then echo "[b8b] MISSING Qwen snapshot" >&2; exit 4; fi
echo "[b8b] ninja=$(command -v ninja || echo NONE) nvcc=$(command -v nvcc || echo NONE) CUDA_HOME=$CUDA_HOME"
command -v ninja >/dev/null || { echo "[b8b] ninja still not on PATH" >&2; exit 4; }
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

loaded () {
  "$PLAN_PY" - "$1" <<'PY' 2>/dev/null || echo 0
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print("1" if (d.get("result") or {}).get("loaded") else "0")
except Exception:
    print("0")
PY
}

N_DOCS_V="${B8_N_DOCS:-50}"
run_vllm () {  # run_vllm <name>
  env -u TORCHDYNAMO_DISABLE -u PYTORCH_CUDA_ALLOC_CONF \
    "$VLLM_PY" "$SCODE/b8_vllm_pilot.py" --backend vllm --name "$1" \
    --tasks "$TASKS" --splits validation --n-docs "$N_DOCS_V" \
    --model-id "Qwen/Qwen3.8-27B" --model-snapshot "$QWEN" \
    --max-new-tokens 160 --max-model-len 1024 \
    --gpu-memory-utilization "${B8_GPU_UTIL:-0.85}" --out "$OUT"
}

echo "[b8b] ===== attempt 1: PATH repaired, vLLM default sampler ====="
set +e; run_vllm vllm_path; echo "[b8b] attempt 1 exit=$?"; set -e

VJSON="$OUT/b8_vllm_path.json"
if [ "$(loaded "$VJSON")" != "1" ]; then
  echo "[b8b] ===== attempt 2: VLLM_USE_FLASHINFER_SAMPLER=0 (no JIT at all) ====="
  export VLLM_USE_FLASHINFER_SAMPLER=0
  set +e; run_vllm vllm_nofi; echo "[b8b] attempt 2 exit=$?"; set -e
  if [ "$(loaded "$OUT/b8_vllm_nofi.json")" = "1" ]; then VJSON="$OUT/b8_vllm_nofi.json"; fi
fi

HFJSON="$OUT/b8_hf.json"
[ -f "$HFJSON" ] || echo "[b8b] WARNING: $HFJSON absent; the verdict will have no controlled denominator"
"$PLAN_PY" "$SCODE/b8_compare.py" --vllm "$VJSON" --hf "$HFJSON" --out "$OUT"
(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS)
plan_done "$OUT"
echo "[b8b] done"
