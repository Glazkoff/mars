#!/bin/bash
# H-J5 -- deployment cost: resident weights, peak GPU memory, batch-1 latency, batched throughput, for MARS-C
# (one seed), FactCG, windowed RoBERTa-MNLI, AlignScore (HF, mars venv) and Bespoke-MiniCheck-7B and the
# Gemma-4-31B judge (vLLM venv, one process each). Registered in research/marsc_ieee_20260922/PREREG.md
# (amendment 3) BEFORE submission; a measurement, no bar.
#
#   sbatch research/marsc_ieee_20260922/slurm/hj5_deploy_cost.sbatch
#SBATCH --job-name=mars-hj5-deploy
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"
JCODE="$PLAN_REPO/research/marsc_ieee_20260922/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
OUT="$MCS/hj5"; mkdir -p "$OUT" "$MCS/logs"; plan_receipt "$OUT"
INV="$M3/m32/units_gemma+ent_validation.jsonl"; [ -f "$INV" ] || { echo "[hj5] MISSING $INV"; exit 4; }
MARSC_CKPT="$MC/e1/mc-A-lam1-crossenc-inv_seed20260906/model"; [ -f "$MARSC_CKPT/config.json" ] || { echo "[hj5] MISSING $MARSC_CKPT"; exit 4; }
FACTCG=$(snap "yaxili96/FactCG-DeBERTa-v3-Large") || true
GEMMA=$(snap "google/gemma-4-31B-it") || true
MINICHECK="${HJ5_MINICHECK_DIR:-/home/user/models/minicheck7b_native}"
MC_UTILS="${HJ5_MINICHECK_UTILS:-/home/user/.venv-mars/lib/python3.12/site-packages/minicheck/utils.py}"
[ -f "$MC_UTILS" ] || { echo "[hj5] MISSING MiniCheck utils $MC_UTILS"; exit 4; }
for p in "$FACTCG" "$GEMMA" "$MINICHECK"; do [ -n "$p" ] && [ -f "$p/config.json" ] || { echo "[hj5] MISSING snapshot $p"; exit 4; }; done
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

echo "===== [hj5] HF systems (mars venv): marsc factcg nli alignscore"
"$PLAN_PY" "$JCODE/hj5_deploy_cost.py" --backend hf --systems marsc factcg nli alignscore \
  --inventory "$INV" --marsc-ckpt "$MARSC_CKPT" --factcg-snapshot "$FACTCG" --out "$OUT" --batch 64

VLLM_PY="${HJ5_VLLM_PY:-/home/user/.venv-vllm/bin/python}"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="/home/user/.venv-vllm/bin:$CUDA_HOME/bin:$PATH"
echo "===== [hj5] MiniCheck-7B (vLLM venv)"
env -u TORCHDYNAMO_DISABLE -u PYTORCH_CUDA_ALLOC_CONF "$VLLM_PY" "$JCODE/hj5_deploy_cost.py" --backend vllm --systems minicheck7b \
  --inventory "$INV" --minicheck-snapshot "$MINICHECK" --minicheck-utils "$MC_UTILS" --max-model-len 8192 --out "$OUT"
echo "===== [hj5] Gemma-4-31B judge (vLLM venv)"
env -u TORCHDYNAMO_DISABLE -u PYTORCH_CUDA_ALLOC_CONF "$VLLM_PY" "$JCODE/hj5_deploy_cost.py" --backend vllm --systems judge \
  --inventory "$INV" --judge-snapshot "$GEMMA" --max-model-len 8192 --out "$OUT"
(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[hj5] done -> $OUT"
