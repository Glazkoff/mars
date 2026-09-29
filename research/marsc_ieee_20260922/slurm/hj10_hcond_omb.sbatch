#!/bin/bash
# H-J10a -- the same-fact crossed candidate-conditioning endpoint on OmissionBench (ComposoAI/OmissionBench,
# CC BY 4.0; synthetic clinical consultations, one fact removed per errored note). Registered in
# research/marsc_ieee_20260922/PREREG.md amendment 7 BEFORE this job was submitted.
#
#   sbatch research/marsc_ieee_20260922/slurm/hj10_hcond_omb.sbatch
#
# Frozen RoSE machinery pointed at OmissionBench docs files: b2_crossed_families.py (its default path
# unchanged; the --window option added for this block is used only in the windowed runs, in their own
# caches), b12_hcond_precompute.py for the three vLLM/HF comparators, and the published bootstrap seed. Two
# docs files (complete omissions = primary; all omission pairs = secondary), two candidate readings each
# (as deployed, 380-word clip; windowed, 380-word windows at stride 190, minimum omission over windows).
#SBATCH --job-name=mars-hj10-hcond
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
JCODE="$PLAN_REPO/research/marsc_ieee_20260922/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
OMB="${HJ10_DATA:-/home/user/data/omissionbench}"
HJ10="$MCS/hj10"; DOCSDIR="$HJ10/docs"; mkdir -p "$DOCSDIR" "$MCS/logs"; plan_receipt "$HJ10"
[ -f "$OMB/pairs/dataset_v2.json" ] && [ -d "$OMB/transcripts" ] || { echo "[hj10] MISSING OmissionBench at $OMB"; exit 4; }

echo "===== [hj10] docs files in the e0 schema (census written before any scorer runs)"
"$PLAN_PY" "$JCODE/hj10_make_docs_omb.py" --pairs "$OMB/pairs/dataset_v2.json" --transcripts "$OMB/transcripts" --out "$DOCSDIR"

HFM="$MC/e1/mc-A-lam1-crossenc-inv_seed*/model"; JFM="$G/b21/xenc_seed*"
HJ9="$MCS/hj9"; NEW_H="$HJ9/models/mc-A-lam1-crossenc-hj9factcghuman_seed*/model"; NEW_J="$HJ9/models/mc-A-lam1-crossenc-hj9factcgjudge_seed*/model"
REG=("humanfact=ckpt:$HFM" "judgefact=ckpt:$JFM" "nli=zs:nli" "summac_zs=zs:summac_zs"
     "lex_token_recall=lex:token_recall" "lex_rougeL_recall=lex:rougeL_recall"
     "emb_minilm=emb:sentence-transformers/all-MiniLM-L6-v2")
EXT=("factcg=zs:factcg" "minicheck7b=zs:minicheck7b" "judge_gemma31=zs:judge_gemma31")
[ "$(ls -d $NEW_H 2>/dev/null | wc -l)" -eq 3 ] && EXT+=("hj9factcghuman=ckpt:$NEW_H") || echo "[hj10] hj9 human arm absent, skipped"
[ "$(ls -d $NEW_J 2>/dev/null | wc -l)" -eq 3 ] && EXT+=("hj9factcgjudge=ckpt:$NEW_J") || echo "[hj10] hj9 judge arm absent, skipped"
ALLFAM=(humanfact judgefact nli summac_zs lex_token_recall lex_rougeL_recall emb_minilm factcg minicheck7b judge_gemma31 hj9factcghuman hj9factcgjudge)

VLLM_PY="${HJ10_VLLM_PY:-/home/user/.venv-vllm/bin/python}"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="/home/user/.venv-vllm/bin:$CUDA_HOME/bin:$PATH"
FACTCG=$(snap "yaxili96/FactCG-DeBERTa-v3-Large") || true
GEMMA=$(snap "google/gemma-4-31B-it") || true
MINICHECK="${HJ10_MINICHECK_DIR:-/home/user/models/minicheck7b_native}"
for p in "$FACTCG" "$GEMMA" "$MINICHECK"; do [ -n "$p" ] && [ -f "$p/config.json" ] || { echo "[hj10] MISSING snapshot $p"; exit 4; }; done

B2 () { "$PLAN_PY" "$SCODE/b2_crossed_families.py" --primary humanfact --batch 64 --n-boot 2000 --seed 20260917 --expect-seeds 3 "$@"; }

for SET in complete all; do
  DOCS="$DOCSDIR/docs_omb_${SET}.jsonl"; [ -s "$DOCS" ] || { echo "[hj10] docs file $DOCS not written"; exit 4; }
  OUT="$HJ10/hcond_${SET}"; CACHE="$OUT/cache"; mkdir -p "$CACHE"; plan_receipt "$OUT"

  echo "===== [hj10] $SET / as deployed: the registered family (b2 default path), primary humanfact"
  B2 --docs "$DOCS" --out "$OUT" --cache "$CACHE" --families "${REG[@]}" 2> >(tee -a "$OUT/b2_stderr.log" >&2) | grep -E "^\[b2\]"

  echo "===== [hj10] $SET: FactCG / MiniCheck-7B / judge into b2's cache (full note, as on UniSumEval)"
  "$PLAN_PY" "$SCODE/b12_hcond_precompute.py" --docs "$DOCS" --system factcg --snapshot "$FACTCG" --family factcg --arg factcg --cache "$CACHE" --batch 32
  env -u TORCHDYNAMO_DISABLE -u PYTORCH_CUDA_ALLOC_CONF "$VLLM_PY" "$SCODE/b12_hcond_precompute.py" --docs "$DOCS" --system minicheck7b \
    --snapshot "$MINICHECK" --family minicheck7b --arg minicheck7b --cache "$CACHE" --trust-remote-code --max-model-len "${HJ10_MC_LEN:-16384}"
  env -u TORCHDYNAMO_DISABLE -u PYTORCH_CUDA_ALLOC_CONF "$VLLM_PY" "$SCODE/b12_hcond_precompute.py" --docs "$DOCS" --system judge \
    --snapshot "$GEMMA" --family judge_gemma31 --arg judge_gemma31 --cache "$CACHE" --max-model-len "${HJ10_JUDGE_LEN:-8192}"

  OUTX="$HJ10/hcond_${SET}_ext"; mkdir -p "$OUTX"; plan_receipt "$OUTX"
  echo "===== [hj10] $SET / as deployed: widened family set (comparators and the H-J9 arms)"
  B2 --docs "$DOCS" --out "$OUTX" --cache "$CACHE" --families "${REG[@]}" "${EXT[@]}" 2> >(tee -a "$OUTX/b2_stderr.log" >&2) | grep -E "^\[b2\]"
  "$PLAN_PY" "$JCODE/hj10_twin_metric.py" --docs "$DOCS" --cache "$CACHE" --families "${ALLFAM[@]}" --out "$HJ10/twin_${SET}.json"

  OUTW="$HJ10/hcond_${SET}_w"; CACHEW="$OUTW/cache"; mkdir -p "$CACHEW"; plan_receipt "$OUTW"
  cp -n "$CACHE"/*.json "$CACHEW"/ 2>/dev/null || true      # the non-ckpt families read the whole note already
  rm -f "$CACHEW"/humanfact.seed*.json "$CACHEW"/judgefact.seed*.json "$CACHEW"/hj9factcg*.seed*.json
  echo "===== [hj10] $SET / windowed (380 words, stride 190, min over windows): registered family"
  B2 --docs "$DOCS" --out "$OUTW" --cache "$CACHEW" --window 380 --stride 190 --families "${REG[@]}" 2> >(tee -a "$OUTW/b2_stderr.log" >&2) | grep -E "^\[b2\]"
  OUTWX="$HJ10/hcond_${SET}_w_ext"; mkdir -p "$OUTWX"; plan_receipt "$OUTWX"
  echo "===== [hj10] $SET / windowed: widened family set"
  B2 --docs "$DOCS" --out "$OUTWX" --cache "$CACHEW" --window 380 --stride 190 --families "${REG[@]}" "${EXT[@]}" 2> >(tee -a "$OUTWX/b2_stderr.log" >&2) | grep -E "^\[b2\]"
  "$PLAN_PY" "$JCODE/hj10_twin_metric.py" --docs "$DOCS" --cache "$CACHEW" --families "${ALLFAM[@]}" --out "$HJ10/twin_${SET}_w.json"
  for d in "$OUT" "$OUTX" "$OUTW" "$OUTWX"; do plan_done "$d"; done
done
(cd "$HJ10" && find . -type f -name "*.json" -not -path "*/cache/*" | sort | xargs sha256sum > SHA256SUMS); plan_done "$HJ10"
echo "[hj10] done -> $HJ10"
