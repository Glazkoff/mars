#!/bin/bash
# MARS-C B5 -- a CONSERVATIVE MACHINE-PROXY LOWER BOUND on the corrected emitter's useful yield.
# Registered: research/marsc_strengthen_20260918/PREREG.md, Wave B / B5.
#
# The paper's only semantic precision evidence annotates an emission the pipeline no longer produces
# (r02_human_rank/emitted_overlap.json: corrected-top-10 overlap 0.5907 / 0.6637 / 0.6322, and only
# 51 / 53 / 48 / 70 of 100 annotated items survive into the corrected top-10). Re-analysing the old labels on
# the survivors is a SELECTION ON RANK, not a measurement, and is not done.
#
# What is defensible with zero new human labels: on 63,343 human-ACU cells the coverage judge's POSITIVE
# omission verdict is near-exact (precision_on_judge_omitted 0.9900) while its recall is only 0.5535 and its
# SUPPORT answer is the measured failure mode (agreement 0.7163, kappa 0.4727). So "judge says omitted" is a
# CONSERVATIVE LOWER BOUND on useful yield and NOTHING ELSE -- never a precision. The judgments for the
# CORRECTED emitter already exist (r07_crossed, 40,586 top-10 tasks, 0 unjudged slots); this job only
# re-aggregates them, clustered on common.doc_key(source) rather than on pair_id.
#
# CPU only. No sealed-TEST re-read: r07_crossed is the VALIDATION emission.
#SBATCH --job-name=mars-b5-yieldlb
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
R7="$MC/r07_crossed"; OUT="$MS/b5b7/b5"; mkdir -p "$OUT"; plan_receipt "$OUT"

# every input is asserted present before anything is computed -- a missing shard would silently shrink the bound
for f in "$R7/tasks.jsonl" "$MC/r04_matched/e0_judge/relabel_report.json" \
         "$MC/r02_human_rank/emitted_overlap.json" "$MC/r02_human_rank/human_precision_ranked.json"; do
  [ -f "$f" ] || { echo "[b5] MISSING $f" >&2; exit 4; }
done
N_SHARDS=$(ls "$R7"/judgments/gemma_coverage_shard*.jsonl 2>/dev/null | wc -l)
[ "$N_SHARDS" -eq 8 ] || { echo "[b5] expected 8 coverage shards, found $N_SHARDS" >&2; exit 4; }
echo "[b5] inputs verified: tasks $(wc -l < "$R7/tasks.jsonl") rows, $N_SHARDS judgment shards"

"$PLAN_PY" "$SCODE/b5_yield_lower_bound.py" \
  --tasks "$R7/tasks.jsonl" \
  --judgments "$R7/judgments/gemma_coverage_shard*.jsonl" \
  --systems "humanfact+div1" "humanfact.distilled+div1" "judgefact+div1" "judgefact.distilled+div1" "mars2" \
  --primary "humanfact+div1" \
  --max-rank 10 --threshold 0.5 --n-boot 5000 --seed 20260918 \
  --judge-validation "$MC/r04_matched/e0_judge/relabel_report.json" \
  --emission-drift "$MC/r02_human_rank/emitted_overlap.json" \
  --human-json "$MC/r02_human_rank/human_precision_ranked.json" \
  --out "$OUT"

(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[b5] done"
