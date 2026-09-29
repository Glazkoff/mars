#!/bin/bash
# H-J3b -- the frozen div1 emission rule, the frozen evaluator and the OBSERVED document-clustered contrasts
# for the three 2026 comparators on UniSumEval, beside the deployed MARS-C arms of record. Registered in
# research/marsc_ieee_20260922/PREREG.md BEFORE submission. Depends on the three hj3_score_unisum jobs:
#
#   sbatch --dependency=afterok:<judge>:<factcg>:<minicheck7b> research/marsc_ieee_20260922/slurm/hj3_eval_unisum.sbatch
#
# A copy of b9b10_eval.sbatch's contract (same rule, evaluator, primary, bootstrap, strict three-seed
# manifests) with the inventory, the ACU file and the deployed rosters pointed at the external pool.
#SBATCH --job-name=mars-hj3-eval
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source "$PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh"
source "$PLAN_REPO/research/mars3_20260916/slurm/env.sh"
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
HJ3="$MCS/hj3"; OUT="$HJ3/eval_unisum"; mkdir -p "$OUT/div"; plan_receipt "$OUT"
INV_U="$MC/g7_unisum/units_gemma+ent_unisum.jsonl"; ACU_U="$MC/g7_unisum/acu_units_unisum.jsonl"; DEP_U="$MC/g7_unisum/val"
for f in "$INV_U" "$ACU_U"; do [ -f "$f" ] || { echo "[hj3e] MISSING $f"; exit 4; }; done

echo "===== [hj3e] the frozen div1 emission rule on every comparator score file"
div () { "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "$1" --scores-glob "$2" --out "$OUT/div"; }
for SYS in judge factcg minicheck7b; do
  SRC="$HJ3/scores_unisum/$SYS/scores"
  [ -d "$SRC" ] || { echo "[hj3e] MISSING scores for $SYS at $SRC"; exit 4; }
  N=$(ls "$SRC"/gemma+ent_*.jsonl 2>/dev/null | wc -l); [ "$N" -ge 3 ] || { echo "[hj3e] $SYS has $N score files, expected 3"; exit 4; }
  div "gemma+ent=$INV_U" "$SRC/gemma+ent_*.jsonl"
done

echo "===== [hj3e] the deployed MARS-C arms of record on UniSumEval (explicit three-seed manifests)"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; \
  [ -f "$f" ] || { echo "[hj3e] MISSING $f" >&2; return 1; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
HF=$(roster "$DEP_U" "gemma+ent_humanfact" "_div1.jsonl") || exit 4
JF=$(roster "$DEP_U" "gemma+ent_xenc" "_div1.jsonl") || exit 4
SPECS="humanfact+div1=$INV_U:$HF judgefact+div1=$INV_U:$JF"
for f in "$OUT"/div/gemma+ent_*_div1.jsonl; do [ -e "$f" ] || continue; b=$(basename "$f" _div1.jsonl); SPECS="$SPECS ${b#gemma+ent_}+div1=$INV_U:$f"; done
echo "===== [hj3e] emitter set:"; for s in $SPECS; do echo "  ${s%%=*}"; done

echo "===== [hj3e] evaluation (k=10; strict three-seed manifests; document-clustered bootstrap)"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU_U" --k 10 --n-boot 2000 \
  --robust-tests --n-perm 10000 --primary "humanfact+div1" --out "$OUT/eval_unisum" \
  --dump-per-pair "$OUT/per_pair_unisum_k10.jsonl" --dump-emitted "$OUT/emitted_unisum_k10.jsonl" \
  --systems $SPECS 2> >(tee -a "$OUT/e2_stderr.log" >&2) | grep -E "\[e2"

echo "===== [hj3e] OBSERVED contrasts with document-clustered intervals (not the bootstrap mean)"
"$PLAN_PY" "$SCODE/b3_contrast.py" --n-boot 2000 --boot-seed 20260918 --skip-missing \
  --eval "unisum_hj3=$OUT/eval_unisum:$OUT/per_pair_unisum_k10.jsonl" --out "$OUT/b3_contrasts.json"

(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[hj3e] done -> $OUT/eval_unisum"
