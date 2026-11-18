#!/bin/bash
# MARS-C R05 -- equivalence / non-inferiority for the label-efficiency curve (FULL_REVIEW item 5). CPU only.
#SBATCH --job-name=mc-r05-equiv
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
OUT="$MC/r05_equivalence"; mkdir -p "$OUT"; plan_receipt "$OUT"
for K in 5 10; do
  echo "===== k=$K"
  "$PLAN_PY" "$MCCODE/mc_equivalence.py" --per-pair "$MC/r01_consensus/per_pair_curve.jsonl" --k "$K" \
    --reference "frac100+div1" --margin 0.05 --n-boot 20000 --out "$OUT/k$K"
done
sha256sum "$OUT"/k*/*.json > "$OUT/SHA256SUMS"; plan_done "$OUT"
