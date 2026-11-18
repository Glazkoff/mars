#!/bin/bash
# MARS-C strengthening C1/C3 -- VALIDATION MECHANICAL DRY RUN (PREREG Wave C / C1, C3).
#
# WHAT THIS JOB IS. It is not a paper number and carries no bar. On validation the campaign has MARS-C's two
# verifier arms and nothing else: r06b_fast ran the off-the-shelf family on TEST only, and r03_sumblind wrote
# the shuffled / no-summary controls on TEST only. So a validation C1 curve has no comparator to dominate and
# a validation C3 fusion has no off-the-shelf constituent to be complementary to. This job therefore exercises
# the whole C1/C3 code path end to end on the split that is not sealed -- rank fusion -> the frozen div1
# emission rule -> mc_e2_eval --k 10 --dump-emitted -> the risk-coverage estimator -- and proves it before
# c1c3_test.sbatch is allowed to touch the held-out split. Its fusion arm (MARS-C humanfact x MARS-C xenc) is
# a code path, not a hypothesis; `dry_run_note` says so inside the output JSON.
#
# If block B1's MiniCheck / AlignScore validation files have landed by the time this runs they are picked up
# automatically and become real comparators; the job prints whether they were found.
#
# Submit:  sbatch research/marsc_strengthen_20260918/slurm/c1c3_validation.sbatch
#SBATCH --job-name=mars-c1c3-val
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source "$PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh"
source "$PLAN_REPO/research/mars3_20260916/slurm/env.sh"
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MCS="${MCS_OUT:-/home/user/results/marsc_strengthen}"
OUT="$MCS/c1c3/validation"; mkdir -p "$OUT/fused" "$OUT/div"; plan_receipt "$OUT"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
INV_GE="$M3/m32/units_gemma+ent_validation.jsonl"; INV_DI="$M3/m32/units_distilled_validation.jsonl"
ACU="$G/b18/acu_units_validation.jsonl"
for f in "$INV_GE" "$INV_DI" "$ACU"; do [ -f "$f" ] || { echo "[c1c3-val] MISSING $f"; exit 4; }; done

roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; \
  [ -f "$f" ] || { echo "[c1c3-val] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }

echo "===== C1C3.v1 raw (pre-div1) validation score manifests for the fusion constituents"
RAW_HF_GE=$(roster "$MC/scores"             "units_gemma+ent_mc-A-lam1-crossenc-inv" ".jsonl")
RAW_HF_DI=$(roster "$MC/e2imp_all/scores"   "distilled_humanfact"                    ".jsonl")
RAW_JF_GE=$(roster "$M3/scores"             "units_gemma+ent_xenc"                   ".jsonl")
RAW_JF_DI=$(roster "$M3/scores"             "units_distilled_xenc"                   ".jsonl")

echo "===== C1C3.v2 rank fusion (rule declared in c1c3_fuse.py before any number is read)"
FU () { "$PLAN_PY" "$SCODE/c1c3_fuse.py" --out "$OUT/fused" --seeds "$SEEDS" "$@"; }
FU --name "fuse.mc_judge"           --unit-set "gemma+ent" --constituent "humanfact=$RAW_HF_GE" --constituent "judgefact=$RAW_JF_GE"
FU --name "fuse.mc_judge.distilled" --unit-set "distilled" --constituent "humanfact=$RAW_HF_DI" --constituent "judgefact=$RAW_JF_DI"

echo "===== C1C3.v3 the frozen div1 emission rule, identical to every other emitter"
"$PLAN_PY" "$MCCODE/mc_diversity.py" --units "gemma+ent=$INV_GE" --scores-glob "$OUT/fused/fuse.mc_judge_seed*.jsonl"           --out "$OUT/div"
"$PLAN_PY" "$MCCODE/mc_diversity.py" --units "distilled=$INV_DI" --scores-glob "$OUT/fused/fuse.mc_judge.distilled_seed*.jsonl" --out "$OUT/div"

echo "===== C1C3.v4 emitter set (MARS-C arms of record + the two fusion code-path arms)"
SPECS="mc.humanfact=$INV_GE:$(roster "$MC/e2div/val"      "units_gemma+ent_mc-A-lam1-crossenc-inv" "_div1.jsonl")"
SPECS="$SPECS mc.humanfact.distilled=$INV_DI:$(roster "$MC/e2div/val" "distilled_humanfact"       "_div1.jsonl")"
SPECS="$SPECS mc.judgefact=$INV_GE:$(roster "$MC/e2div_full/val" "units_gemma+ent_xenc"           "_div1.jsonl")"
SPECS="$SPECS mc.judgefact.distilled=$INV_DI:$(roster "$MC/e2div/val" "units_distilled_xenc"      "_div1.jsonl")"
SPECS="$SPECS fuse.mc_judge=$INV_GE:$(roster "$OUT/div" "fuse.mc_judge"                           "_div1.jsonl")"
SPECS="$SPECS fuse.mc_judge.distilled=$INV_DI:$(roster "$OUT/div" "fuse.mc_judge.distilled"       "_div1.jsonl")"

COMPARATORS=""
B1DIV="$MCS/b1/validation/div"
if [ -d "$B1DIV" ]; then
  for f in "$B1DIV"/gemma+ent_*_div1.jsonl; do [ -e "$f" ] || continue; b=$(basename "$f" _div1.jsonl); \
    SPECS="$SPECS zs.${b#gemma+ent_}=$INV_GE:$f"; COMPARATORS="$COMPARATORS --comparator zs.${b#gemma+ent_}"; done
  for f in "$B1DIV"/distilled_*_div1.jsonl;  do [ -e "$f" ] || continue; b=$(basename "$f" _div1.jsonl); \
    SPECS="$SPECS zs.distilled.${b#distilled_}=$INV_DI:$f"; COMPARATORS="$COMPARATORS --comparator zs.distilled.${b#distilled_}"; done
fi
if [ -z "$COMPARATORS" ]; then
  echo "[c1c3-val] NOTE: no block-B1 validation verifier files under $B1DIV -- this run has NO off-the-shelf"
  echo "[c1c3-val]       comparator and no summary-blind control on validation, exactly as the header says."
else
  echo "[c1c3-val] NOTE: block-B1 validation verifier files FOUND, they enter as comparators:$COMPARATORS"
fi
echo "===== C1C3.v5 emitters:"; for s in $SPECS; do echo "  ${s%%=*}"; done

echo "===== C1C3.v6 evaluation (--k 10 alone, so --dump-emitted fires at k=10)"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU" --k 10 --n-boot 2000 \
  --primary "mc.humanfact" --out "$OUT/eval_validation" \
  --dump-emitted "$OUT/emitted_validation_k10.jsonl" --systems $SPECS 2>&1 | grep -E "\[e2"

echo "===== C1C3.v7 risk-coverage estimator (C1) and complementarity (C3), document-clustered"
"$PLAN_PY" "$SCODE/c1c3_riskcov.py" --emitted "$OUT/emitted_validation_k10.jsonl" --acu "$ACU" \
  --out "$OUT/analysis" --n-boot 2000 --k-label 10 \
  --label "validation MECHANICAL DRY RUN" \
  --dry-run-note "Validation carries no off-the-shelf comparator and no summary-blind control (r06b_fast and r03_sumblind wrote TEST only), so no C1 or C3 bar is evaluated here and no number in this file is a paper number. This run exists to prove the C1/C3 code path before the sealed-TEST job." \
  --primary mc.humanfact --primary mc.humanfact.distilled $COMPARATORS \
  --fusion "fuse.mc_judge=mc.humanfact+mc.judgefact" \
  --fusion "fuse.mc_judge.distilled=mc.humanfact.distilled+mc.judgefact.distilled" \
  --eval-json "$OUT/eval_validation/e2_emitted.json" 2>&1 | grep -E "\[c1|\[c3"

(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[c1c3-val] done -> $OUT/analysis/c1c3_result.json"
