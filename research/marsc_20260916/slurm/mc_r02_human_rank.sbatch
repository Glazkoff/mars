#!/bin/bash
# MARS-C R02 -- rank-recovered human precision (FULL_REVIEW_2026-09-17 decision-critical failure 2). CPU only.
#SBATCH --job-name=mc-r02-humanrank
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
OUT="$MC/r02_human_rank"; mkdir -p "$OUT"; plan_receipt "$OUT"
"$PLAN_PY" "$MCCODE/mc_human_rank.py" \
  --emitted "$MC/g3/emitted_validation.jsonl" \
  --key "$MC/g6/human_sample_key_all.jsonl" \
  --export "$MC/g6/labels_export_all.jsonl" \
  --e2-json "$MC/g3/eval_validation/e2_emitted.json" \
  --budgets 10 20 --n-boot 10000 --out "$OUT" 2>&1 | tee "$OUT/r02.log"
sha256sum "$OUT"/*.json > "$OUT/SHA256SUMS"; plan_done "$OUT"
