#!/bin/bash
# MARS-C R02b -- overlap between the annotated emission and the corrected one, and the precision estimate
# recomputed against the CORRECTED emitter and the CORRECTED status weights. CPU only.
#SBATCH --job-name=mc-r02b-overlap
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
OUT="$MC/r02_human_rank"; mkdir -p "$OUT"
echo "===== R02b.1 overlap of the annotated top-10 with the corrected top-10"
"$PLAN_PY" "$MCCODE/mc_emitted_overlap.py" --as-annotated "$MC/g3/emitted_validation.jsonl" \
  --corrected "$MC/r07_crossed/emitted_validation_k10.jsonl" --key "$MC/g6/human_sample_key_all.jsonl" \
  --k 10 --out "$OUT"
echo "===== R02b.2 precision against the CORRECTED emitter and CORRECTED status weights"
"$PLAN_PY" "$MCCODE/mc_human_rank.py" \
  --emitted "$MC/r07_crossed/emitted_validation_k10.jsonl" \
  --key "$MC/g6/human_sample_key_all.jsonl" --export "$MC/g6/labels_export_all.jsonl" \
  --e2-json "$MC/r07_crossed/eval_validation/e2_emitted.json" --budgets 10 --n-boot 10000 --out "$OUT/corrected"
echo "===== R02b.3 the as-annotated estimate, re-weighted by the CORRECTED status mix"
"$PLAN_PY" "$MCCODE/mc_human_rank.py" \
  --emitted "$MC/g3/emitted_validation.jsonl" \
  --key "$MC/g6/human_sample_key_all.jsonl" --export "$MC/g6/labels_export_all.jsonl" \
  --e2-json "$MC/r07_crossed/eval_validation/e2_emitted.json" --budgets 10 20 --n-boot 10000 --out "$OUT/reweighted"
