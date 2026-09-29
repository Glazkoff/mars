#!/bin/bash
# MARS-C B9/B10 evaluation -- apply the frozen div1 emission rule to the LLM-judge and 2025/26-leader verifier
# arms and evaluate them BESIDE the MARS-C arms of record, on the split named by E_SPLIT.
#
# This is the step that turns score files into the rows MARS_PUBLICATION_PLAN_2026-09-20.md Sec. 3.2 asks for:
# recall@10 and omission precision on the emitted endpoint. It is a copy of b1_eval.sbatch's contract with the
# score directories swapped -- same emission rule, same evaluator, same primary, same bootstrap -- so a B9/B10
# row is not distinguishable downstream from a B1 row or from an arm of record.
#
# Reporting rule, inherited from b1_eval.sbatch and frozen before any of these numbers was read: ALL completed
# rows enter the main table, ordered by recall@10, with omission precision and the summary-blind controls
# beside them, WHATEVER the ordering turns out to be.
#
#   sbatch --export=ALL,E_SPLIT=validation research/marsc_strengthen_20260918/slurm/b9b10_eval.sbatch
#   sbatch --export=ALL,E_SPLIT=test,MC_TEST_REREAD=yes research/marsc_strengthen_20260918/slurm/b9b10_eval.sbatch
#
#SBATCH --job-name=mars-b9b10-eval
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
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
SPLIT="${E_SPLIT:-validation}"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
[ "$SPLIT" = "validation" ] || [ "$SPLIT" = "test" ] || { echo "[e] bad E_SPLIT=$SPLIT"; exit 4; }
if [ "$SPLIT" = "test" ]; then
  [ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[e] REFUSED: sealed TEST split needs MC_TEST_REREAD=yes"; exit 3; }
fi

# E_INVS restricts which inventories enter this pass ("distilled", "gemma+ent", or both). An early pass over
# a finished inventory must not glob one that is still being written, so the restriction is explicit.
INVS="${E_INVS:-distilled gemma+ent}"
TAG="${E_TAG:-}"
OUT="$MCS/b9b10_eval/$SPLIT$TAG"; mkdir -p "$OUT/div"; plan_receipt "$OUT"
INV_GE="$M3/m32/units_gemma+ent_${SPLIT}.jsonl"; INV_DI="$M3/m32/units_distilled_${SPLIT}.jsonl"
ACU="$G/b18/acu_units_${SPLIT}.jsonl"
for f in "$INV_GE" "$INV_DI" "$ACU"; do [ -f "$f" ] || { echo "[e] MISSING $f"; exit 4; }; done

echo "===== the frozen div1 emission rule, identical to every other emitter"
div () { "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "$1" --scores-glob "$2" --out "$OUT/div"; }
for SRC in "$MCS/b9/$SPLIT/scores" "$MCS/b10/$SPLIT/scores"; do
  [ -d "$SRC" ] || { echo "[e] no scores at $SRC (skipped)"; continue; }
  for IV in $INVS; do
    case "$IV" in
      distilled)  U="$INV_DI" ;;
      gemma+ent)  U="$INV_GE" ;;
      *) echo "[e] unknown inventory $IV"; exit 4 ;;
    esac
    ls "$SRC"/${IV}_*.jsonl >/dev/null 2>&1 && div "$IV=$U" "$SRC/${IV}_*.jsonl"
  done
done

echo "===== the MARS-C arms of record on this split (explicit three-seed manifests)"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; \
  [ -f "$f" ] || { echo "[e] MISSING $f" >&2; return 1; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
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
for f in "$OUT"/div/gemma+ent_*_div1.jsonl; do [ -e "$f" ] || continue; b=$(basename "$f" _div1.jsonl); SPECS="$SPECS ${b#gemma+ent_}+div1=$INV_GE:$f"; done
for f in "$OUT"/div/distilled_*_div1.jsonl;  do [ -e "$f" ] || continue; b=$(basename "$f" _div1.jsonl); SPECS="$SPECS distilled.${b#distilled_}+div1=$INV_DI:$f"; done
echo "===== emitter set:"; for s in $SPECS; do echo "  ${s%%=*}"; done

echo "===== evaluation (--k 10 alone, so --dump-emitted fires at k=10)"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU" --k 10 --n-boot 2000 \
  --robust-tests --n-perm 10000 --primary "humanfact+div1" --out "$OUT/eval_${SPLIT}" \
  --dump-per-pair "$OUT/per_pair_${SPLIT}_k10.jsonl" --dump-emitted "$OUT/emitted_${SPLIT}_k10.jsonl" \
  --systems $SPECS 2> >(tee -a "$OUT/e2_stderr.log" >&2) | grep -E "\[e2"
(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[e] done -> $OUT/eval_${SPLIT}"
