#!/bin/bash
# H-J3a -- the same-fact crossed candidate-conditioning endpoint on UniSumEval, for the registered family
# and, in a separate block, the three post-registration comparators. Registered in
# research/marsc_ieee_20260922/PREREG.md BEFORE this job was submitted.
#
#   sbatch research/marsc_ieee_20260922/slurm/hj3_hcond_unisum.sbatch
#
# Everything is the frozen RoSE machinery pointed at a UniSumEval docs file: b2_crossed_families.py and
# mc_summary_conditioning.py are imported unmodified, b12_hcond_precompute.py writes the three new arms into
# b2's own cache exactly as it did for RoSE, and the bootstrap seed is the published one.
#SBATCH --job-name=mars-hj3-hcond
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:30:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
JCODE="$PLAN_REPO/research/marsc_ieee_20260922/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
HJ3="$MCS/hj3"; DOCSDIR="$HJ3/docs"; OUT="$HJ3/hcond_unisum"; OUTX="$HJ3/hcond_unisum_ext"; CACHE="$OUT/cache"
mkdir -p "$DOCSDIR" "$CACHE" "$OUTX" "$MCS/logs"; plan_receipt "$OUT"

ACU="$MC/g7_unisum/acu_units_unisum.jsonl"; [ -f "$ACU" ] || { echo "[hj3] MISSING $ACU"; exit 4; }
echo "===== [hj3] docs file in the e0 schema (census written before any scorer runs)"
"$PLAN_PY" "$JCODE/hj3_make_docs_unisum.py" --acu "$ACU" --out "$DOCSDIR"
DOCS="$DOCSDIR/docs_unisum.jsonl"; [ -s "$DOCS" ] || { echo "[hj3] docs file not written"; exit 4; }

HFM="$MC/e1/mc-A-lam1-crossenc-inv_seed*/model"; JFM="$G/b21/xenc_seed*"
echo "===== [hj3] the registered family (b2, unmodified), primary humanfact"
"$PLAN_PY" "$SCODE/b2_crossed_families.py" --docs "$DOCS" --out "$OUT" --cache "$CACHE" \
  --primary humanfact --batch 64 --n-boot 2000 --seed 20260917 --expect-seeds 3 --families \
  "humanfact=ckpt:$HFM" "judgefact=ckpt:$JFM" "nli=zs:nli" "summac_zs=zs:summac_zs" \
  "lex_token_recall=lex:token_recall" "lex_rougeL_recall=lex:rougeL_recall" \
  "emb_minilm=emb:sentence-transformers/all-MiniLM-L6-v2" \
  2> >(tee -a "$OUT/b2_stderr.log" >&2) | grep -E "^\[b2\]"

VLLM_PY="${HJ3_VLLM_PY:-/home/user/.venv-vllm/bin/python}"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="/home/user/.venv-vllm/bin:$CUDA_HOME/bin:$PATH"
FACTCG=$(snap "yaxili96/FactCG-DeBERTa-v3-Large") || true
GEMMA=$(snap "google/gemma-4-31B-it") || true
MINICHECK="${HJ3_MINICHECK_DIR:-/home/user/models/minicheck7b_native}"
for p in "$FACTCG" "$GEMMA" "$MINICHECK"; do
  [ -n "$p" ] && [ -f "$p/config.json" ] || { echo "[hj3] MISSING snapshot $p"; exit 4; }
done

echo "===== [hj3] FactCG (HF cross-encoder, mars venv) into b2's cache"
"$PLAN_PY" "$SCODE/b12_hcond_precompute.py" --docs "$DOCS" --system factcg --snapshot "$FACTCG" \
  --family factcg --arg factcg --cache "$CACHE" --batch 32

echo "===== [hj3] Bespoke-MiniCheck-7B (vLLM venv) into b2's cache"
env -u TORCHDYNAMO_DISABLE -u PYTORCH_CUDA_ALLOC_CONF \
  "$VLLM_PY" "$SCODE/b12_hcond_precompute.py" --docs "$DOCS" --system minicheck7b --snapshot "$MINICHECK" \
  --family minicheck7b --arg minicheck7b --cache "$CACHE" --trust-remote-code \
  --max-model-len "${HJ3_MC_LEN:-32768}"

echo "===== [hj3] Gemma-4-31B judge (vLLM venv, exact score mode) into b2's cache"
env -u TORCHDYNAMO_DISABLE -u PYTORCH_CUDA_ALLOC_CONF \
  "$VLLM_PY" "$SCODE/b12_hcond_precompute.py" --docs "$DOCS" --system judge --snapshot "$GEMMA" \
  --family judge_gemma31 --arg judge_gemma31 --cache "$CACHE" --max-model-len "${HJ3_JUDGE_LEN:-8192}"

echo "===== [hj3] b2 over the widened family set (the three new arms served from the cache)"
"$PLAN_PY" "$SCODE/b2_crossed_families.py" --docs "$DOCS" --cache "$CACHE" \
  --primary humanfact --batch 64 --n-boot 2000 --seed 20260917 --expect-seeds 3 --out "$OUTX" \
  --families "humanfact=ckpt:$HFM" "judgefact=ckpt:$JFM" "nli=zs:nli" "summac_zs=zs:summac_zs" \
    "lex_token_recall=lex:token_recall" "lex_rougeL_recall=lex:rougeL_recall" \
    "emb_minilm=emb:sentence-transformers/all-MiniLM-L6-v2" \
    "factcg=zs:factcg" "minicheck7b=zs:minicheck7b" "judge_gemma31=zs:judge_gemma31" \
  2> >(tee -a "$OUTX/b2_stderr.log" >&2) | grep -E "^\[b2\]"

(cd "$HJ3" && find docs hcond_unisum hcond_unisum_ext -type f -name "*.json" | sort | xargs sha256sum > "$OUT/SHA256SUMS")
plan_done "$OUT"; plan_done "$OUTX"
echo "[hj3] done -> $OUT/crossed_families.json and $OUTX/crossed_families.json"
