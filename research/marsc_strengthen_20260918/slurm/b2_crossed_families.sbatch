#!/bin/bash
# MARS-C strengthen B2 -- H-COND: the same-fact crossed-accuracy endpoint for EVERY verifier family.
#
# The campaign's primary new hypothesis. Within a document, for the SAME fact, does the verifier score the fact
# higher against a summary that OMITS it than against a summary that CONVEYS it? A summary-blind scorer is
# pinned at exactly 0.500, so unlike recall@k this endpoint cannot be won by ignoring the candidate -- which is
# the defect the paper's own shuffled-summary control exposed (recall@10 rises 0.377 -> 0.440 shuffled -> 0.458
# with no summary at all).
#
# Families: MARS-C humanfact (3 seeds), MARS-C judge-label xenc (3 seeds), windowed NLI, SummaC-ZS, and the
# lexical / MiniLM counters. The counters are the implementation's sanity check: they read the candidate through
# a bag of tokens or one embedding, so a bug that dropped the candidate entirely would pin every family at 0.500.
#
# Bars are declared in the PREREG and written into the output JSON BEFORE the numbers, so the record shows they
# preceded the result. Submit:
#   sbatch --time=02:00:00 --export=ALL,B2_SPLIT=validation  research/marsc_strengthen_20260918/slurm/b2_crossed_families.sbatch
#   sbatch --time=03:00:00 --export=ALL,B2_SPLIT=test,MC_TEST_REREAD=yes  <same>
#SBATCH --job-name=mars-b2-crossed
#SBATCH --partition=train
#SBATCH --nodelist=Euler
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
SPLIT="${B2_SPLIT:-validation}"
case "$SPLIT" in validation|test) ;; *) echo "[b2] bad B2_SPLIT=$SPLIT"; exit 2 ;; esac
if [ "$SPLIT" = "test" ]; then
  [ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[b2] REFUSED: sealed TEST re-read without MC_TEST_REREAD=yes"; exit 3; }
fi
DOCS="$MC/e0/docs_${SPLIT}.jsonl"; [ -f "$DOCS" ] || { echo "[b2] MISSING $DOCS"; exit 4; }
HFM="$MC/e1/mc-A-lam1-crossenc-inv_seed*/model"; JFM="$G/b21/xenc_seed*"
OUT="$MCS/b2/crossed_${SPLIT}"; mkdir -p "$OUT/cache"; plan_receipt "$OUT"
# The per-arm cache makes this job RESUMABLE: chain two or three copies with --dependency=afterany
# and each backfill window advances the arms it can finish. Delete $OUT/cache to force a rescore.
echo "[b2] split=$SPLIT docs=$DOCS out=$OUT"

"$PLAN_PY" "$SCODE/b2_crossed_families.py" --docs "$DOCS" --out "$OUT" \
  --cache "$OUT/cache" \
  --primary humanfact --batch 64 --n-boot 2000 --seed 20260917 --expect-seeds 3 --families \
  "humanfact=ckpt:$HFM" \
  "judgefact=ckpt:$JFM" \
  "nli=zs:nli" \
  "summac_zs=zs:summac_zs" \
  "lex_token_recall=lex:token_recall" \
  "lex_rougeL_recall=lex:rougeL_recall" \
  "emb_minilm=emb:sentence-transformers/all-MiniLM-L6-v2"

# The registered bar is the CONJUNCTION over validation and the sealed TEST split; combine whatever is final.
HAVE=""
for S in validation test; do
  F="$MCS/b2/crossed_${S}/crossed_families.json"
  if [ -f "$F" ]; then HAVE="$HAVE $F"; fi
done
if [ -n "$HAVE" ]; then
  "$PLAN_PY" "$SCODE/b2_crossed_families.py" --combine $HAVE --out "$MCS/b2"
fi
(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[b2] done -> $OUT"
