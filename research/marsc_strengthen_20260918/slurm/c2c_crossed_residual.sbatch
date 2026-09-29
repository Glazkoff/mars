#!/bin/bash
# MARS-C strengthen C2 / H-SURF (c) -- the registered BAR.
#
# "The MARS-C advantage on the B2 crossed-conditioning endpoint must survive residualisation with its interval
# still excluding zero."  This job adjudicates exactly that sentence, on validation AND on the sealed TEST split.
#
# It scores the same-fact crossed cells with the same checkpoints and the same scoring contract R03a/B2 use
# (mc_summary_conditioning.py, imported unmodified), dumps the per-cell z so the endpoint never has to be
# re-derived from a GPU again, fits the three registered surface regressors on the VALIDATION cells per
# checkpoint seed, applies them UNCHANGED to the TEST cells, and recomputes the endpoint on the residual.
#
# Reported beside the bar, because it is the honest reading of it: a SURFACE-ONLY arm (the fitted prediction
# used as the score).  Inside a same-fact triple the fact is held fixed, so unit position and unit length are
# identical in the two cells and cannot move the win indicator; the only surface channel that can fake
# conditioning on this endpoint is fact-summary lexical overlap, and the surface-only arm measures exactly how
# far that channel alone gets.
#SBATCH --job-name=mars-c2c-crossed-resid
#SBATCH --partition=train
#SBATCH --nodelist=Euler
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[c2c] REFUSED -- sealed TEST re-read needs MC_TEST_REREAD=yes"; exit 3; }
OUT="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}/c2c4"
JOB="$OUT/c2c_crossed"; mkdir -p "$JOB"; plan_receipt "$JOB"

DOCS_V="$MC/e0/docs_validation.jsonl"; DOCS_T="$MC/e0/docs_test.jsonl"
for f in "$DOCS_V" "$DOCS_T"; do [ -f "$f" ] || { echo "[c2c] MISSING $f" >&2; exit 4; }; done
# the two checkpoint families R03a/B2 compare; both are rescorable (weights on disk), verified before submission
HF_CK="$MC/e1/mc-A-lam1-crossenc-inv_seed*/model"
JF_CK="$G/b21/xenc_seed*"

"$PLAN_PY" "$SCODE/c2_crossed_residual.py" \
  --docs-validation "$DOCS_V" --docs-test "$DOCS_T" \
  --families "humanfact=$HF_CK" "judgefact=$JF_CK" \
  --primary humanfact --batch 64 --n-boot 2000 --seed 20260917 --expect-seeds 3 \
  --out "$JOB" 2>&1 | grep -E "^\[c2x"

(cd "$OUT" && find c2c_crossed -type f -name "*.json" | sort | xargs sha256sum > c2c_crossed/SHA256SUMS)
plan_done "$JOB"
echo "[c2c] done -> $JOB"
