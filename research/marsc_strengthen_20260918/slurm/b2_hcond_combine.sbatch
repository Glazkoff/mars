#!/bin/bash
# MARS-C strengthen B2 -- the H-COND conjunction over validation and the sealed TEST split.
#
# The registered bar is a CONJUNCTION: MARS-C must clear every off-the-shelf family on BOTH splits. The two
# crossed-accuracy jobs are submitted independently so each can backfill on its own, so the conjunction is
# taken here, CPU-only, once both have written their crossed_families.json. Reads nothing but those two files.
#   sbatch --dependency=afterany:<val>:<test> research/marsc_strengthen_20260918/slurm/b2_hcond_combine.sbatch
#SBATCH --job-name=mars-b2-hcond
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
HAVE=""
for S in validation test; do
  F="$MCS/b2/crossed_${S}/crossed_families.json"
  if [ -f "$F" ]; then HAVE="$HAVE $F"; else echo "[b2c] absent: $F"; fi
done
[ -n "$HAVE" ] || { echo "[b2c] neither split produced crossed_families.json"; exit 4; }
OUT="$MCS/b2"; mkdir -p "$OUT"; plan_receipt "$OUT"
"$PLAN_PY" "$SCODE/b2_crossed_families.py" --combine $HAVE --out "$OUT"
(cd "$OUT" && find . -maxdepth 1 -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[b2c] done -> $OUT/hcond_verdict.json"
