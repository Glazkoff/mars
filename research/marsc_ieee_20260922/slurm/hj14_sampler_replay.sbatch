#!/bin/bash
# H-J14 (2026-09-29, registered in research/marsc_ieee_20260922/PREREG.md, Amendment H-J14): replay the training
# sampler of the R04 twins under both label sets and record whether the two runs drew the same cells. CPU only.
#SBATCH --job-name=mars-hj14-replay
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
MC="${MARSC_OUT:-/home/user/results/marsc}"
OUT="/home/user/results/marsc_strengthen/hj14"; mkdir -p "$OUT"; plan_receipt "$OUT"
"$PLAN_PY" "$PLAN_REPO/research/marsc_ieee_20260922/code/hj14_sampler_replay.py" \
  --human "$MC/e0" --judge "$MC/r04_matched/e0_judge" --models "$MC/r04_matched/models" --out "$OUT"
(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[hj14] done -> $OUT"
