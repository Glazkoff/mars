#!/bin/bash
# MARS-C B6a -- the CPU half of the stage-level cost account, plus a PROVISIONAL amortisation curve.
# Registered: research/marsc_strengthen_20260918/PREREG.md, Wave B / B6. No bar: a cost measurement cannot
# pass or fail, the measured number is the number.
#
# Why this is split from the GPU half: the node's eight GPUs are fully allocated and a GPU job queues, while
# the three stages the paper has NEVER priced are all CPU work and can land immediately --
#   * the emission rule (mc_diversity.py re-segments every source and maps every unit onto a sentence, then
#     walks the ranking applying the -5.0 demotion) -- paid on EVERY summary, priced nowhere;
#   * the spaCy entity pass, which is the "+ent" half of the gemma+ent inventory and is absent from the G4
#     cost table, so the paper's 6.637 s/document prices only the LLM half of its own inventory of record;
#   * claim rendering, which the off-the-shelf comparator family pays and MARS-C does not.
# It also re-derives the two population constants the G4 table took from a 200-pair window: 165.5 units/pair
# (split-wide 94) and 11.76 pairs/document. Every per-pair second in the published cost paragraph inherits
# that window, so the split-wide counts are computed here and the curve is priced at them.
#
# The curve this job writes is PROVISIONAL: verifier and decomposition rates fall back to the recorded G4
# numbers, each labelled `declared` in the output. b6b_cost_gpu.sbatch overwrites nothing -- it writes its own
# stage file and its own curve (amortisation_curve.json) from measured rates.
#
# CPU only. Validation inventories only; the sealed TEST split is not read.
#SBATCH --job-name=mars-b6a-cost-cpu
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
OUT="$MS/b6b8/b6"; mkdir -p "$OUT"; plan_receipt "$OUT"

INV_GE="$M3/m32/units_gemma+ent_validation.jsonl"
INV_DI="$M3/m32/units_distilled_validation.jsonl"
for f in "$INV_GE" "$INV_DI" "$MC/g4/cost_table.json"; do
  [ -f "$f" ] || { echo "[b6a] MISSING $f" >&2; exit 4; }
done
echo "[b6a] inventories: gemma+ent $(wc -l < "$INV_GE") rows, distilled $(wc -l < "$INV_DI") rows"

"$PLAN_PY" "$SCODE/b6_cost.py" --stage timings --which cpu \
  --inv-gemma "$INV_GE" --inv-distilled "$INV_DI" \
  --n-pairs "${B6_N_PAIRS:-200}" --n-docs "${B6_N_DOCS:-50}" \
  --timings-name stage_times_cpu.json --out "$OUT"

"$PLAN_PY" "$SCODE/b6_cost.py" --stage curve \
  --timings-name stage_times_cpu.json --curve-name curve_provisional.json --out "$OUT"

(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS_cpu)
plan_done "$OUT"
echo "[b6a] done"
