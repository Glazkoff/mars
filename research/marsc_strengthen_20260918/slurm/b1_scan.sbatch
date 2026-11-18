#!/bin/bash
# MARS-C strengthening B1.1 -- the sharded off-the-shelf verifier scan (PREREG Wave B / B1 + the B0 standing
# control rule). One array task = one (inventory, candidate-mode, shard) block of one split, scored by every
# verifier named in B1_SYSTEMS. Shard blocks are contiguous row ranges balanced on unit count; the row offset
# of every block is printed here and recorded in the task's b1_manifest_*.json.
#
# Submit (validation first, then the sealed TEST split):
#   sbatch --array=0-5%2 --export=ALL,B1_SPLIT=validation,B1_SYSTEMS=minicheck+alignscore ...
#   sbatch --array=0-8%2 --export=ALL,B1_SPLIT=test,MC_TEST_REREAD=yes,B1_SYSTEMS=minicheck+alignscore ...
# Sizing (pilot job 5251, 200 spread units of the validation distilled inventory, one idle H200):
#   MiniCheck fast path 131.8 units/s -- and max|delta| 7.90e-07 against the official package on the same
#   200 units, 0 units over 1e-3, 0 rank disagreements, so the fast path IS the official arm;
#   AlignScore 666.0 units/s (previously UNMEASURED anywhere in this project).
#   Combined 110 units/s => the largest block below (150180 units) is ~23 min on an idle GPU.
#SBATCH --job-name=mars-b1-scan
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source "$PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh"
source "$PLAN_REPO/research/mars3_20260916/slurm/env.sh"
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
ROOT="${MCS_OUT:-/home/user/results/marsc_strengthen}/b1"
SPLIT="$B1_SPLIT"                         # set by --export; unset is a hard error under `set -u`
# `sbatch --export` splits its value on commas, so B1_SYSTEMS/B1_MODES use "+" as the
# separator: `--export=ALL,B1_SYSTEMS=minicheck,alignscore` silently sets B1_SYSTEMS=minicheck
# and exports a variable called `alignscore` (verified, job 5305).
SYSTEMS="${B1_SYSTEMS:-minicheck+alignscore}"; SYSTEMS="${SYSTEMS//+/,}"
[ "$SPLIT" = "validation" ] || [ "$SPLIT" = "test" ] || { echo "[b1-scan] bad B1_SPLIT=$SPLIT"; exit 4; }
if [ "$SPLIT" = "test" ]; then
  [ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[b1-scan] REFUSED: sealed TEST split needs MC_TEST_REREAD=yes"; exit 3; }
fi

# shard counts, sized from the pilot so no block exceeds ~25 min on an idle GPU and ~75 min under contention:
#   val/distilled 58592, val/gemma+ent 138261, test/distilled 123576, test/gemma+ent 300360
INVS=(distilled gemma+ent)
if [ "$SPLIT" = "validation" ]; then NSH=(1 1); else NSH=(1 2); fi
TASKS=()
for MODE in real shuffled empty; do
  for ii in 0 1; do
    n=${NSH[$ii]}
    for (( s=1; s<=n; s++ )); do TASKS+=("${INVS[$ii]} $MODE $s $n"); done
  done
done
echo "===== B1.1 task table for split=$SPLIT (${#TASKS[@]} tasks)"
for i in "${!TASKS[@]}"; do echo "  [$i] ${TASKS[$i]}"; done
T="${SLURM_ARRAY_TASK_ID:-0}"
[ "$T" -lt "${#TASKS[@]}" ] || { echo "[b1-scan] array index $T >= ${#TASKS[@]} tasks"; exit 4; }
read -r INV MODE SH_I SH_N <<< "${TASKS[$T]}"
UNITS="$M3/m32/units_${INV}_${SPLIT}.jsonl"
[ -f "$UNITS" ] || { echo "[b1-scan] MISSING $UNITS"; exit 4; }
OUT="$ROOT/$SPLIT/parts"; mkdir -p "$OUT"; plan_receipt "$ROOT/$SPLIT/receipts/task${T}_${INV}_${MODE}_${SH_I}of${SH_N}"
echo "===== B1.1 task $T: split=$SPLIT inv=$INV mode=$MODE shard=$SH_I/$SH_N systems=$SYSTEMS"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
"$PLAN_PY" "$SCODE/b1_verifier_scan.py" --unit-set m2 --units "$UNITS" --units-name "$INV" --split "$SPLIT" \
  --systems "$SYSTEMS" --candidate-mode "$MODE" --shard "$SH_I/$SH_N" --batch 64 --chunk 512 --out "$OUT"
echo "[b1-scan] task $T done"
