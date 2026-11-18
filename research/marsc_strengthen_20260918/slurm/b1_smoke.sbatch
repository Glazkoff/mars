#!/bin/bash
# MARS-C strengthening B1.0b -- a 3-pair smoke of the scan/shard/merge path that the pilot did not exercise
# (the pilot only ran --verify-canonical). Writes into a throwaway directory; nothing here is a result.
#SBATCH --job-name=mars-b1-smoke
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source "$PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh"
source "$PLAN_REPO/research/mars3_20260916/slurm/env.sh"
export MC="${MARSC_OUT:-/home/user/results/marsc}"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
OUT="${MCS_OUT:-/home/user/results/marsc_strengthen}/b1/smoke"; rm -rf "$OUT"; mkdir -p "$OUT/parts"; plan_receipt "$OUT"
U="$M3/m32/units_distilled_validation.jsonl"
SYS="${B1_SYSTEMS:-minicheck,alignscore,nli,summac_zs,lex,minilm}"
for MODE in real shuffled empty; do
  for S in 1 2 3; do
    "$PLAN_PY" "$SCODE/b1_verifier_scan.py" --unit-set m2 --units "$U" --units-name distilled \
      --split validation --systems "$SYS" --candidate-mode "$MODE" --shard "$S/3" --limit 2 \
      --batch 64 --chunk 512 --out "$OUT/parts" 2>&1 | grep -E "\[b1\]"
  done
done
echo "===== smoke: parts written"; ls -1 "$OUT/parts" | head -60
echo "===== smoke: the merge MUST refuse here -- --limit 2 dropped most pairs from every shard"
set +e
"$PLAN_PY" "$SCODE/b1_verifier_scan.py" --merge --unit-set m2 --units "$U" --units-name distilled \
  --split validation --systems "$SYS" --modes real,shuffled,empty --parts "$OUT/parts" --out "$OUT/scores" 2>&1 | tail -5
RC=$?; set -e
[ "$RC" -ne 0 ] && echo "[b1-smoke] merge refused as designed (rc=$RC)" || { echo "[b1-smoke] FAIL: merge accepted an incomplete part set"; exit 5; }
echo "===== smoke: now a complete 3-shard pass with no --limit, which MUST merge"
rm -rf "$OUT/parts"; mkdir -p "$OUT/parts"
for S in 1 2 3; do
  "$PLAN_PY" "$SCODE/b1_verifier_scan.py" --unit-set m2 --units "$U" --units-name distilled \
    --split validation --systems lex --candidate-mode real --shard "$S/3" \
    --batch 64 --chunk 512 --out "$OUT/parts" 2>&1 | grep -E "\[b1\]"
done
"$PLAN_PY" "$SCODE/b1_verifier_scan.py" --merge --unit-set m2 --units "$U" --units-name distilled \
  --split validation --systems lex --modes real --parts "$OUT/parts" --out "$OUT/scores"
plan_done "$OUT"; echo "[b1-smoke] done"
