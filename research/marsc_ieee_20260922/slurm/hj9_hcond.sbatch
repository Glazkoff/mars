#!/bin/bash
# H-J9 -- the crossed candidate-conditioning endpoint (H-COND) for the new FactCG-encoder arms on both pools,
# with every comparator family served from the caches the published rows used (b12/test for RoSE, hj3 for
# UniSumEval), so the estimator, triples, bootstrap and margins are b2's own. Registered in PREREG.md
# amendment 6.
#
#   sbatch --dependency=afterok:<human>:<judge> --export=ALL,MC_TEST_REREAD=yes research/marsc_ieee_20260922/slurm/hj9_hcond.sbatch
#SBATCH --job-name=mars-hj9-hcond
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[hj9c] REFUSED: needs MC_TEST_REREAD=yes"; exit 3; }
HJ9="${HJ9_ROOT:-$MCS/hj9}"; BB="${HJ9_BACKBONE:-factcg}"
HFM="$MC/e1/mc-A-lam1-crossenc-inv_seed*/model"; JFM="$G/b21/xenc_seed*"
NEW_H="$HJ9/models/mc-A-lam1-crossenc-hj9${BB}human_seed*/model"; NEW_J="$HJ9/models/mc-A-lam1-crossenc-hj9${BB}judge_seed*/model"
for POOL in test unisum; do
  case "$POOL" in
    test)   DOCS="$MC/e0/docs_test.jsonl"; PREV="$MCS/b12/test/cache" ;;
    unisum) DOCS="$MCS/hj3/docs/docs_unisum.jsonl"; PREV="$MCS/hj3/hcond_unisum/cache" ;;
  esac
  [ -f "$DOCS" ] || { echo "[hj9c] MISSING $DOCS"; exit 4; }
  OUT="$HJ9/hcond_${POOL}"; CACHE="$OUT/cache"; mkdir -p "$CACHE"; plan_receipt "$OUT"
  cp -n "$PREV"/*.json "$CACHE"/ 2>/dev/null || true
  echo "[hj9c] $POOL: seeded cache with $(ls -1 "$CACHE" | wc -l) files from $PREV"
  FAMS=("humanfact=ckpt:$HFM" "judgefact=ckpt:$JFM" "nli=zs:nli" "summac_zs=zs:summac_zs"
        "lex_token_recall=lex:token_recall" "lex_rougeL_recall=lex:rougeL_recall"
        "emb_minilm=emb:sentence-transformers/all-MiniLM-L6-v2"
        "factcg=zs:factcg" "minicheck7b=zs:minicheck7b" "judge_gemma31=zs:judge_gemma31"
        "hj9${BB}human=ckpt:$NEW_H")
  ls -d $NEW_J >/dev/null 2>&1 && FAMS+=("hj9${BB}judge=ckpt:$NEW_J")
  "$PLAN_PY" "$SCODE/b2_crossed_families.py" --docs "$DOCS" --cache "$CACHE" \
    --primary "hj9${BB}human" --batch 64 --n-boot 2000 --seed 20260917 --expect-seeds 3 --out "$OUT" \
    --families "${FAMS[@]}" 2> >(tee -a "$OUT/b2_stderr.log" >&2) | grep -E "^\[b2\]"
  (cd "$OUT" && find . -maxdepth 1 -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
done
echo "[hj9c] done"
