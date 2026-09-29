#!/bin/bash
# H-J11 (gap fill, 2026-09-27) -- the same-fact crossed endpoint on RoSE VALIDATION for the four encoder arms
# of H-J9/H-J9b, which were scored only on TEST and UniSumEval. The validation comparator cache of record
# (b12/validation, jobs 5850/5851) is copied, never written, so every existing family is served from the
# SAME scores the table already uses and only the twelve new checkpoint-seeds are computed. --primary humanfact
# gives the table's "MARS-C minus row" margins directly. Validation is not the sealed split.
#SBATCH --job-name=mars-hj11-hcond-val
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%j.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
DOCS="$MC/e0/docs_validation.jsonl"; [ -f "$DOCS" ] || { echo "[hj11] MISSING $DOCS"; exit 4; }
OUT="$MCS/hj11/hcond_validation"; CACHE="$OUT/cache"; mkdir -p "$CACHE"; plan_receipt "$OUT"
cp -n "$MCS/b12/validation/cache/"*.json "$CACHE/"; echo "[hj11] seeded cache: $(ls -1 "$CACHE" | wc -l) files"
"$PLAN_PY" "$SCODE/b2_crossed_families.py" --docs "$DOCS" --cache "$CACHE" \
  --primary humanfact --batch 64 --n-boot 2000 --seed 20260917 --expect-seeds 3 --out "$OUT" \
  --families \
    "humanfact=ckpt:$MC/e1/mc-A-lam1-crossenc-inv_seed*/model" \
    "judgefact=ckpt:$G/b21/xenc_seed*" \
    "nli=zs:nli" "summac_zs=zs:summac_zs" \
    "lex_token_recall=lex:token_recall" "lex_rougeL_recall=lex:rougeL_recall" \
    "emb_minilm=emb:sentence-transformers/all-MiniLM-L6-v2" \
    "factcg=zs:factcg" "minicheck7b=zs:minicheck7b" "judge_gemma31=zs:judge_gemma31" \
    "hj9factcghuman=ckpt:$MCS/hj9/models/mc-A-lam1-crossenc-hj9factcghuman_seed*/model" \
    "hj9factcgjudge=ckpt:$MCS/hj9/models/mc-A-lam1-crossenc-hj9factcgjudge_seed*/model" \
    "hj9xlargehuman=ckpt:$MCS/hj9_xlarge/models/mc-A-lam1-crossenc-hj9xlargehuman_seed*/model" \
    "hj9xlargejudge=ckpt:$MCS/hj9_xlarge/models/mc-A-lam1-crossenc-hj9xlargejudge_seed*/model" \
  2> >(tee -a "$OUT/b2_stderr.log" >&2) | grep -E "^\[b2\]"
plan_done "$OUT"; echo "[hj11] done -> $OUT"
