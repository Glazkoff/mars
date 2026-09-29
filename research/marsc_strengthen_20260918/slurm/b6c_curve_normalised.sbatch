#!/bin/bash
# MARS-C B6c -- the corpus-normalised amortisation curve, and the audit of the two published decomposition
# seconds-per-document. Registered: research/marsc_strengthen_20260918/PREREG.md, Wave B / B6. No bar.
#
# WHY A THIRD JOB. B6b measured the decomposition stages on 50 (Gemma: 25) RoSE validation documents, which
# carry 27.7--31.0 sentences each against the frozen task file's corpus average of 19.21. Seconds per SENTENCE
# is the invariant rate; seconds per DOCUMENT read straight off that sample is not the corpus figure, and
# pricing the curve at it would repeat exactly the sample-window mistake this block was written to correct
# (the G4 table's 165.5 units/pair against a split-wide 94.3). This job re-prices the decomposition stages at
# measured s/sentence x corpus sentences/document, leaving B6b's sample-rate curve untouched as
# amortisation_curve.json and writing the corpus-normalised one as amortisation_curve_corpus.json.
#
# It also runs the denominator audit against results/mars3/m32/distill_result.json, which records that the
# distil job trained on 27,207 sentences and predicted 10,951 -- so the published 0.2367 s/document divides a
# figure containing a one-off TRAINING cost by 1,986 documents the segmenter never predicted for.
#
# CPU only, seconds of work, reads only artifacts this block already wrote plus two frozen files.
#SBATCH --job-name=mars-b6c-curve-norm
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
OUT="$MS/b6b8/b6"
TASKS="$M3/m32/decompose_tasks.jsonl"; DR="$M3/m32/distill_result.json"
for f in "$OUT/stage_times_gpu.json" "$OUT/stage_times_cpu.json" "$TASKS" "$DR"; do
  [ -f "$f" ] || { echo "[b6c] MISSING $f" >&2; exit 4; }
done

"$PLAN_PY" "$SCODE/b6_cost.py" --stage curve \
  --timings-name stage_times_gpu.json --merge "$OUT/stage_times_cpu.json" \
  --tasks "$TASKS" --distill-result "$DR" \
  --curve-name amortisation_curve_corpus.json --out "$OUT"

(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS)
plan_done "$OUT"
echo "[b6c] done"
