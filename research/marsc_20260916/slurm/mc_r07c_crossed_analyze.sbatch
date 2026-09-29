#!/bin/bash
# MARS-C R07c -- the crossed 2x2 decomposition of emitted-fact precision.
#SBATCH --job-name=mc-r07c-an
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
OUT="$MC/r07_crossed"
"$PLAN_PY" "$MCCODE/mc_crossed_precision.py" --tasks "$OUT/tasks.jsonl" \
  --judgments "$OUT/judgments/gemma_coverage_shard*.jsonl" --max-rank 10 --n-boot 5000 \
  --cells "gemma+ent:human=humanfact+div1" "gemma+ent:judge=judgefact+div1" \
          "distilled:human=humanfact.distilled+div1" "distilled:judge=judgefact.distilled+div1" \
  --extra-systems mars2 --human-json "$MC/r02_human_rank/human_precision_ranked.json" --out "$OUT/analysis"
sha256sum "$OUT"/analysis/*.json > "$OUT/SHA256SUMS"
