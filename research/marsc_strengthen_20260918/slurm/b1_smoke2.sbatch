#!/bin/bash
# MARS-C strengthening B1.0c -- the single-shard (n=1) scan->merge path, which is exactly the validation
# configuration and which the 3-shard smoke (job 5268) did not cover. CPU only: `lex` needs no GPU.
#SBATCH --job-name=mars-b1-smoke2
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source "$PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh"
source "$PLAN_REPO/research/mars3_20260916/slurm/env.sh"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
OUT="${MCS_OUT:-/home/user/results/marsc_strengthen}/b1/smoke2"; rm -rf "$OUT"; mkdir -p "$OUT/parts"; plan_receipt "$OUT"
U="$M3/m32/units_distilled_validation.jsonl"
for MODE in real shuffled empty; do
  "$PLAN_PY" "$SCODE/b1_verifier_scan.py" --unit-set m2 --units "$U" --units-name distilled --split validation \
    --systems lex --candidate-mode "$MODE" --shard "1/1" --device cpu --out "$OUT/parts" 2>&1 | grep -E "\[b1\]"
done
echo "===== smoke2: parts"; ls -1 "$OUT/parts"
"$PLAN_PY" "$SCODE/b1_verifier_scan.py" --merge --unit-set m2 --units "$U" --units-name distilled \
  --split validation --systems lex --modes real,shuffled,empty --parts "$OUT/parts" --out "$OUT/scores"
echo "===== smoke2: merged"; ls -1 "$OUT/scores"
N=$(ls -1 "$OUT/scores"/*.jsonl | wc -l)
[ "$N" -eq 6 ] || { echo "[b1-smoke2] FAIL: expected 6 merged files (2 lex names x 3 modes), got $N"; exit 5; }
echo "===== smoke2: the real and empty premises must give DIFFERENT scores"
"$PLAN_PY" - "$OUT/scores" <<'PY'
import sys, json
from pathlib import Path
d = Path(sys.argv[1])
a = [json.loads(l) for l in open(d / "distilled_lex_token_recall.jsonl")]
b = [json.loads(l) for l in open(d / "distilled_lex_token_recall_empty.jsonl")]
c = [json.loads(l) for l in open(d / "distilled_lex_token_recall_shuf.jsonl")]
assert [x["pair_id"] for x in a] == [x["pair_id"] for x in b] == [x["pair_id"] for x in c]
na = sum(1 for x in a for v in x["scores"] if v is not None and v < 1.0)
nb = sum(1 for x in b for v in x["scores"] if v is not None and v < 1.0)
diff = sum(1 for x, y in zip(a, c) for u, v in zip(x["scores"], y["scores"]) if u != v)
tot = sum(len(x["scores"]) for x in a)
print(f"[smoke2] units={tot} real-with-any-overlap={na} empty-with-any-overlap={nb} real!=shuffled={diff}")
assert nb == 0, "the empty premise must give omission score 1.0 everywhere for a lexical counter"
assert na > 0 and diff > 0.5 * tot, "real / shuffled premises are not actually different"
print("[smoke2] premise modes behave as registered")
PY
plan_done "$OUT"; echo "[b1-smoke2] done"
