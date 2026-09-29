#!/bin/bash
# MARS-C strengthening B1.2 -- merge the sharded scores, apply the frozen div1 emission rule, and evaluate
# every verifier row beside the MARS-C arms of record on the split named by B1_SPLIT.
#
# Reporting rule, frozen in PREREG.md before any of these numbers was read: ALL completed verifier rows enter
# the main table, ordered by recall@10, with omission precision and the summary-blind controls (candidate-mode
# shuffled / empty) beside them, WHATEVER the ordering turns out to be. SummaC-ZS already leads MARS-C on
# recall@10 on TEST (0.3848 vs 0.3765); MiniCheck or AlignScore may too. That outcome is published.
#
# Submit:
#   sbatch --export=ALL,B1_SPLIT=validation,B1_SYSTEMS=minicheck+alignscore+lex+minilm+nli+summac_zs ...
#   sbatch --export=ALL,B1_SPLIT=test,MC_TEST_REREAD=yes,B1_SYSTEMS=... (same list) ...
#SBATCH --job-name=mars-b1-eval
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source "$PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh"
source "$PLAN_REPO/research/mars3_20260916/slurm/env.sh"
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
ROOT="${MCS_OUT:-/home/user/results/marsc_strengthen}/b1"
SPLIT="$B1_SPLIT"
# `sbatch --export` splits its value on commas, so B1_SYSTEMS/B1_MODES use "+" as the
# separator: `--export=ALL,B1_SYSTEMS=minicheck,alignscore` silently sets B1_SYSTEMS=minicheck
# and exports a variable called `alignscore` (verified, job 5305).
SYSTEMS="${B1_SYSTEMS:-minicheck+alignscore}"; SYSTEMS="${SYSTEMS//+/,}"
MODES="${B1_MODES:-real+shuffled+empty}"; MODES="${MODES//+/,}"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
[ "$SPLIT" = "validation" ] || [ "$SPLIT" = "test" ] || { echo "[b1-eval] bad B1_SPLIT=$SPLIT"; exit 4; }
if [ "$SPLIT" = "test" ]; then
  [ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[b1-eval] REFUSED: sealed TEST split needs MC_TEST_REREAD=yes"; exit 3; }
fi
OUT="$ROOT/$SPLIT"; mkdir -p "$OUT/scores" "$OUT/div"; plan_receipt "$OUT"
INV_GE="$M3/m32/units_gemma+ent_${SPLIT}.jsonl"; INV_DI="$M3/m32/units_distilled_${SPLIT}.jsonl"
ACU="$G/b18/acu_units_${SPLIT}.jsonl"
for f in "$INV_GE" "$INV_DI" "$ACU"; do [ -f "$f" ] || { echo "[b1-eval] MISSING $f"; exit 4; }; done

echo "===== B1.2a merge the shards (refuses unless every pair is present with its inventory unit count)"
for INV in distilled gemma+ent; do
  U="$M3/m32/units_${INV}_${SPLIT}.jsonl"
  "$PLAN_PY" "$SCODE/b1_verifier_scan.py" --merge --unit-set m2 --units "$U" --units-name "$INV" \
    --split "$SPLIT" --systems "$SYSTEMS" --modes "$MODES" --parts "$OUT/parts" --out "$OUT/scores"
done

echo "===== B1.2b the frozen div1 emission rule, identical to every other emitter"
div () { "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "$1" --scores-glob "$2" --out "$OUT/div"; }
div "gemma+ent=$INV_GE" "$OUT/scores/gemma+ent_*.jsonl"
div "distilled=$INV_DI"  "$OUT/scores/distilled_*.jsonl"

echo "===== B1.2c the MARS-C arms of record on this split (explicit three-seed manifests)"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; \
  [ -f "$f" ] || { echo "[b1-eval] MISSING $f" >&2; return 1; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
if [ "$SPLIT" = "test" ]; then
  G1="$MC/g1_test"
  HF_GE=$(roster "$G1/val" "gemma+ent_humanfact" "_div1.jsonl")
  HF_DI=$(roster "$G1/val" "distilled_humanfact" "_div1.jsonl")
  JF_GE=$(roster "$G1/val" "gemma+ent_xenc" "_div1.jsonl")
  JF_DI=$(roster "$G1/val" "distilled_xenc" "_div1.jsonl")
else
  DIV="$MC/e2div"; DF="$MC/e2div_full"
  HF_GE=$(roster "$DIV/val" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_div1.jsonl")
  HF_DI=$(roster "$DIV/val" "distilled_humanfact" "_div1.jsonl")
  JF_GE=$(roster "$DF/val"  "units_gemma+ent_xenc" "_div1.jsonl")
  JF_DI=$(roster "$DIV/val" "units_distilled_xenc" "_div1.jsonl")
fi
SPECS="humanfact+div1=$INV_GE:$HF_GE humanfact.distilled+div1=$INV_DI:$HF_DI \
judgefact+div1=$INV_GE:$JF_GE judgefact.distilled+div1=$INV_DI:$JF_DI"
for f in "$OUT"/div/gemma+ent_*_div1.jsonl; do b=$(basename "$f" _div1.jsonl); SPECS="$SPECS zs.${b#gemma+ent_}+div1=$INV_GE:$f"; done
for f in "$OUT"/div/distilled_*_div1.jsonl;  do b=$(basename "$f" _div1.jsonl); SPECS="$SPECS zs.distilled.${b#distilled_}+div1=$INV_DI:$f"; done
echo "===== B1.2d emitter set:"; for s in $SPECS; do echo "  ${s%%=*}"; done

echo "===== B1.2e evaluation (--k 10 alone, so --dump-emitted fires at k=10)"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU" --k 10 --n-boot 2000 \
  --robust-tests --n-perm 10000 --primary "humanfact+div1" --out "$OUT/eval_${SPLIT}" \
  --dump-per-pair "$OUT/per_pair_${SPLIT}_k10.jsonl" --dump-emitted "$OUT/emitted_${SPLIT}_k10.jsonl" \
  --systems $SPECS 2>&1 | grep -E "\[e2"
(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[b1-eval] done -> $OUT/eval_${SPLIT}"
