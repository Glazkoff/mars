#!/bin/bash
# MARS-C B12 -- H-COND for the three arms added this week, then B2's own verdict over the widened family set.
# MARS_PUBLICATION_PLAN_2026-09-20.md §3.2: the judge row is owed "recall@10, omission precision and H-COND".
#
#   sbatch --export=ALL,B12_SPLIT=validation research/marsc_strengthen_20260918/slurm/b12_hcond.sbatch
#   sbatch --export=ALL,B12_SPLIT=test,MC_TEST_REREAD=yes research/marsc_strengthen_20260918/slurm/b12_hcond.sbatch
#
#SBATCH --job-name=mars-b12-hcond
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
SPLIT="${B12_SPLIT:-validation}"
if [ "$SPLIT" = "test" ]; then
  [ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[b12] REFUSED: sealed TEST needs MC_TEST_REREAD=yes"; exit 3; }
fi
DOCS="$MC/e0/docs_${SPLIT}.jsonl"
[ -f "$DOCS" ] || { echo "[b12] MISSING $DOCS"; exit 4; }

OUT="$MCS/b12/$SPLIT"; CACHE="$OUT/cache"; mkdir -p "$CACHE" "$MCS/logs"; plan_receipt "$OUT"
# Seed from B2's existing cache so humanfact/judgefact/nli/summac_zs are served from the SAME scores the
# published H-COND table used, and only the three new arms are computed. The original cache is copied, never
# written to, so the earlier run stays reproducible.
PREV="$MCS/b2/crossed_${SPLIT}/cache"
if [ -d "$PREV" ]; then cp -n "$PREV"/*.json "$CACHE"/ 2>/dev/null || true; echo "[b12] seeded cache from $PREV ($(ls -1 "$CACHE" | wc -l) files)"; fi
VLLM_PY="${B12_VLLM_PY:-/home/user/.venv-vllm/bin/python}"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0        # the internlm2 forward patch must reach the engine
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="/home/user/.venv-vllm/bin:$CUDA_HOME/bin:$PATH"

# `snap` ends in `head -1`; under `set -euo pipefail` an empty result makes the assignment the failing
# command and the job dies with a bare exit code, so the MISSING-snapshot messages below never print.
# `|| true` keeps the diagnostics reachable.
FACTCG=$(snap "yaxili96/FactCG-DeBERTa-v3-Large") || true
GEMMA=$(snap "google/gemma-4-31B-it") || true
MINICHECK="${B12_MINICHECK_DIR:-/home/user/models/minicheck7b_native}"
for p in "$FACTCG" "$GEMMA" "$MINICHECK"; do
  [ -n "$p" ] && [ -f "$p/config.json" ] || { echo "[b12] MISSING snapshot $p"; exit 4; }
done
echo "[b12] split=$SPLIT docs=$DOCS cache=$CACHE"

echo "===== [b12] FactCG (HF cross-encoder, mars venv)"
"$PLAN_PY" "$SCODE/b12_hcond_precompute.py" --docs "$DOCS" --system factcg --snapshot "$FACTCG" \
  --family factcg --arg factcg --cache "$CACHE" --batch 32

echo "===== [b12] Bespoke-MiniCheck-7B (vLLM venv)"
env -u TORCHDYNAMO_DISABLE -u PYTORCH_CUDA_ALLOC_CONF \
  "$VLLM_PY" "$SCODE/b12_hcond_precompute.py" --docs "$DOCS" --system minicheck7b --snapshot "$MINICHECK" \
  --family minicheck7b --arg minicheck7b --cache "$CACHE" --trust-remote-code \
  --max-model-len "${B12_MC_LEN:-32768}"

echo "===== [b12] Gemma-4-31B judge (vLLM venv)"
env -u TORCHDYNAMO_DISABLE -u PYTORCH_CUDA_ALLOC_CONF \
  "$VLLM_PY" "$SCODE/b12_hcond_precompute.py" --docs "$DOCS" --system judge --snapshot "$GEMMA" \
  --family judge_gemma31 --arg judge_gemma31 --cache "$CACHE" --max-model-len "${B12_JUDGE_LEN:-4096}"

echo "===== [b12] B2's verdict over the widened family set (every new arm served from the cache)"
# The lex/emb counter families are carried over from b2_crossed_families.sbatch deliberately: they read the
# candidate only through a bag of tokens or one embedding, so they are expected NEAR but not AT 0.500 and a
# bug that dropped the candidate entirely would pin every family at exactly 0.500. Dropping them would
# remove the implementation's own candidate-blindness sanity check AND change the comparator set the
# registered H-COND verdict is stated over.
# The three new families are declared as zs:<arg>; b2's cached() reports a HIT and never builds a scorer, so
# the estimator, triples, bootstrap and margins are b2's own, unmodified.
"$PLAN_PY" "$SCODE/b2_crossed_families.py" --docs "$DOCS" --cache "$CACHE" \
  --primary humanfact --batch 64 --n-boot 2000 --seed 20260917 --expect-seeds 3 --out "$OUT" \
  --families \
    "humanfact=ckpt:$MC/e1/mc-A-lam1-crossenc-inv_seed*/model" \
    "judgefact=ckpt:$G/b21/xenc_seed*" \
    "nli=zs:nli" "summac_zs=zs:summac_zs" \
    "lex_token_recall=lex:token_recall" "lex_rougeL_recall=lex:rougeL_recall" \
    "emb_minilm=emb:sentence-transformers/all-MiniLM-L6-v2" \
    "factcg=zs:factcg" "minicheck7b=zs:minicheck7b" "judge_gemma31=zs:judge_gemma31" \
  2> >(tee -a "$OUT/b2_stderr.log" >&2) | grep -E "^\[b2\]"

plan_done "$OUT"
echo "[b12] done -> $OUT"
