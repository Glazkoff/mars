#!/bin/bash
# H-J3c -- the overlap-residualised crossed endpoint (H-SURF's estimator) applied to UniSumEval.
# Registered as an amendment in research/marsc_ieee_20260922/PREREG.md BEFORE submission: post-hoc
# secondary, no bar. The surface model is fitted on the RoSE VALIDATION cells per checkpoint seed exactly as
# in C2, and applied unchanged to the UniSumEval crossed cells; c2_crossed_residual.py is used unmodified,
# with its --docs-test argument pointed at the external docs file.
#
#   sbatch research/marsc_ieee_20260922/slurm/hj3_surf_unisum.sbatch
#SBATCH --job-name=mars-hj3-surf
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
OUT="$MCS/hj3/surf_unisum"; mkdir -p "$OUT"; plan_receipt "$OUT"
DOCS_V="$MC/e0/docs_validation.jsonl"; DOCS_U="$MCS/hj3/docs/docs_unisum.jsonl"
for f in "$DOCS_V" "$DOCS_U"; do [ -f "$f" ] || { echo "[hj3s] MISSING $f"; exit 4; }; done
HF_CK="$MC/e1/mc-A-lam1-crossenc-inv_seed*/model"; JF_CK="$G/b21/xenc_seed*"
"$PLAN_PY" "$SCODE/c2_crossed_residual.py" \
  --docs-validation "$DOCS_V" --docs-test "$DOCS_U" \
  --families "humanfact=$HF_CK" "judgefact=$JF_CK" \
  --primary humanfact --batch 64 --n-boot 2000 --seed 20260917 --expect-seeds 3 \
  --out "$OUT" 2>&1 | grep -E "^\[c2x"
(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[hj3s] done -> $OUT"
