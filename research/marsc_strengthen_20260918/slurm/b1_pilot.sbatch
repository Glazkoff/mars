#!/bin/bash
# MARS-C strengthening B1.0 -- the pilot that has to run before any full MiniCheck/AlignScore array.
# Three jobs in one, all on 200 units spread over the validation distilled inventory:
#   (1) print the file-level evidence that the R06 chain never produced a MiniCheck or an AlignScore row;
#   (2) measure the official MiniCheck path against the batched re-implementation and REFUSE if they differ
#       by more than 1e-3 (the full scan is only defensible if the fast path is the official one);
#   (3) measure AlignScore units/s, which is UNMEASURED anywhere in this project, so that b1_scan.sbatch
#       can be sized instead of guessed.
#SBATCH --job-name=mars-b1-pilot
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source "$PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh"
source "$PLAN_REPO/research/mars3_20260916/slurm/env.sh"
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
OUT="${MCS_OUT:-/home/user/results/marsc_strengthen}/b1/pilot"; mkdir -p "$OUT"; plan_receipt "$OUT"

echo "===== B1.0 evidence: what the R06 chain actually produced"
echo "--- r06_offshelf (job 5001, CANCELLED) scores:"; ls -1 "$MC/r06_offshelf/scores" 2>/dev/null || echo "  (none)"
echo "--- r06_offshelf val/ (should be empty):"; ls -1A "$MC/r06_offshelf/val" 2>/dev/null | wc -l
echo "--- r06_offshelf complete.json:"; ls -1 "$MC/r06_offshelf/complete.json" 2>/dev/null || echo "  ABSENT -- the job never completed"
echo "--- r06b_fast (job 5035, COMPLETE) scores:"; ls -1 "$MC/r06b_fast/scores"
echo "--- any minicheck/alignscore score file anywhere under results/marsc:"
find "$MC" -name "*minicheck*" -o -name "*alignscore*" 2>/dev/null | head -20 || true
echo "--- last line of the cancelled r06 log:"; tail -1 "$MC/logs/mc-r06-offshelf-5001.out" 2>/dev/null || true

echo "===== B1.0 asset check (a metadata-only stub resolves to a path and then fails at load)"
du -sh "$HF_HOME/hub/models--yzha--AlignScore" "$HF_HOME/hub/models--lytang--MiniCheck-Flan-T5-Large" \
       "$HF_HOME/hub/models--roberta-large"
ls -l "$HF_HOME/hub/models--yzha--AlignScore/snapshots"/*/AlignScore-large.ckpt

INV_DI_V="$M3/m32/units_distilled_validation.jsonl"
echo "===== B1.0 pilot: 200 spread units of $INV_DI_V"
"$PLAN_PY" "$SCODE/b1_verifier_scan.py" --unit-set m2 --units "$INV_DI_V" --units-name distilled \
  --split validation --verify-canonical 200 --verify-tol 1e-3 --batch 64 --out "$OUT"

(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[b1-pilot] done"
