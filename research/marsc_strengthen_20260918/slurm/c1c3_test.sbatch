#!/bin/bash
# MARS-C strengthening C1 (H-SEL) + C3 (H-COMP) on the sealed TEST split (PREREG Wave C / C1, C3).
#
# WHY THIS JOB IS ON TEST AND NOT ON VALIDATION. C1 needs comparator families on the same axes and C3 needs an
# off-the-shelf constituent to fuse with. Both exist only on TEST: r06b_fast scored lex / minilm / windowed NLI /
# SummaC-ZS over the two frozen inventories on TEST, and r03_sumblind wrote the shuffled and no-summary controls
# on TEST. c1c3_validation.sbatch proves the code path on the unsealed split first; this job is submitted with
# --dependency=afterok on it. PREREG.md is the dated amendment authorising the TEST re-read.
#
# WHAT IS DECLARED BEFORE ANY NUMBER IS READ (and is written into the output JSON before the numbers):
#   * the fusion rule -- mean of per-pair normalised ranks, ties averaged -- in c1c3_fuse.py;
#   * the C1 bar -- MARS-C dominates a comparator iff the document-clustered 95% interval of the paired
#     precision difference is above zero at EVERY coverage grid point in [0.20, 1.00];
#   * the C3 bar -- a fusion passes iff at coverage 1.00 the document-clustered 95% interval of the paired
#     omission-precision difference excludes zero from above against BOTH of its constituents;
#   * the choice of fusion partner. The strongest off-the-shelf family is SummaC-ZS, chosen from the ALREADY
#     FROZEN r06b_fast TEST table (recall@10 0.3848 distilled / 0.3630 gemma+ent, the highest of the four
#     completed families and higher than MARS-C's own 0.3765), not from anything this job computes.
#     nli_nli is fused as the second-strongest partner and lex_token_recall as a DECLARED NULL CONTROL
#     fusion: if fusing anything at all raises precision, lex must raise it too and the C3 claim is vacuous.
#   * the B0 control fusions. The standing campaign rule is that every new emitter arm ships with its
#     shuffled-summary and no-summary controls. A fusion is a new arm, so SummaC-ZS is also fused with the
#     shuffled-summary and no-summary MARS-C scores. If those fusions gain as much, the gain is not
#     complementarity with a candidate-reading verifier and C3 is reported as refuted whatever the bar says.
#
# Block B1 (MiniCheck / AlignScore) files are picked up automatically if they have landed; the job prints
# whether they were found and they enter the C1 comparator set when they are.
#
# Submit:  sbatch --export=ALL,MC_TEST_REREAD=yes research/marsc_strengthen_20260918/slurm/c1c3_test.sbatch
#SBATCH --job-name=mars-c1c3-test
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source "$PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh"
source "$PLAN_REPO/research/mars3_20260916/slurm/env.sh"
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MCS="${MCS_OUT:-/home/user/results/marsc_strengthen}"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[c1c3-test] REFUSED: sealed TEST split needs MC_TEST_REREAD=yes"; exit 3; }
OUT="$MCS/c1c3/test"; mkdir -p "$OUT/fused" "$OUT/div"; plan_receipt "$OUT"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
G1="$MC/g1_test"; R3="$MC/r03_sumblind"; R6="$MC/r06b_fast"
INV_GE="$M3/m32/units_gemma+ent_test.jsonl"; INV_DI="$M3/m32/units_distilled_test.jsonl"
ACU="$G/b18/acu_units_test.jsonl"
for f in "$INV_GE" "$INV_DI" "$ACU"; do [ -f "$f" ] || { echo "[c1c3-test] MISSING $f"; exit 4; }; done

roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; \
  [ -f "$f" ] || { echo "[c1c3-test] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
need () { [ -f "$1" ] || { echo "[c1c3-test] MISSING $1"; exit 4; }; printf '%s' "$1"; }

echo "===== C1C3.t1 raw (pre-div1) TEST manifests for the fusion constituents"
RAW_HF_GE=$(roster "$G1/scores" "gemma+ent_humanfact" ".jsonl");        RAW_HF_DI=$(roster "$G1/scores" "distilled_humanfact" ".jsonl")
RAW_SH_GE=$(roster "$R3/scores" "gemma+ent_humanfactshuffled" ".jsonl"); RAW_SH_DI=$(roster "$R3/scores" "distilled_humanfactshuffled" ".jsonl")
RAW_EM_GE=$(roster "$R3/scores" "gemma+ent_humanfactempty" ".jsonl");    RAW_EM_DI=$(roster "$R3/scores" "distilled_humanfactempty" ".jsonl")
RAW_SUMMAC_GE=$(need "$R6/scores/gemma+ent_nli_summac_zs.jsonl");  RAW_SUMMAC_DI=$(need "$R6/scores/distilled_nli_summac_zs.jsonl")
RAW_NLI_DI=$(need "$R6/scores/distilled_nli_nli.jsonl");           RAW_LEX_DI=$(need "$R6/scores/distilled_lex_token_recall.jsonl")

echo "===== C1C3.t2 rank fusion (rule declared in c1c3_fuse.py before any number is read)"
FU () { "$PLAN_PY" "$SCODE/c1c3_fuse.py" --out "$OUT/fused" --seeds "$SEEDS" "$@"; }
FU --name "fuse.mc_summac.distilled"      --unit-set distilled --constituent "humanfact=$RAW_HF_DI" --constituent "summac=$RAW_SUMMAC_DI"
FU --name "fuse.mc_nli.distilled"         --unit-set distilled --constituent "humanfact=$RAW_HF_DI" --constituent "nli=$RAW_NLI_DI"
FU --name "fuse.mc_lex.distilled"         --unit-set distilled --constituent "humanfact=$RAW_HF_DI" --constituent "lex=$RAW_LEX_DI"
FU --name "fuse.mcshuf_summac.distilled"  --unit-set distilled --constituent "humanfactshuffled=$RAW_SH_DI" --constituent "summac=$RAW_SUMMAC_DI"
FU --name "fuse.mcempty_summac.distilled" --unit-set distilled --constituent "humanfactempty=$RAW_EM_DI"    --constituent "summac=$RAW_SUMMAC_DI"
FU --name "fuse.mc_summac"                --unit-set "gemma+ent" --constituent "humanfact=$RAW_HF_GE" --constituent "summac=$RAW_SUMMAC_GE"
FU --name "fuse.mcshuf_summac"            --unit-set "gemma+ent" --constituent "humanfactshuffled=$RAW_SH_GE" --constituent "summac=$RAW_SUMMAC_GE"
FU --name "fuse.mcempty_summac"           --unit-set "gemma+ent" --constituent "humanfactempty=$RAW_EM_GE"    --constituent "summac=$RAW_SUMMAC_GE"

echo "===== C1C3.t3 the frozen div1 emission rule, identical to every other emitter"
"$PLAN_PY" "$MCCODE/mc_diversity.py" --units "distilled=$INV_DI" --scores-glob "$OUT/fused/fuse.*.distilled_seed*.jsonl" --out "$OUT/div"
for n in fuse.mc_summac fuse.mcshuf_summac fuse.mcempty_summac; do
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "gemma+ent=$INV_GE" --scores-glob "$OUT/fused/${n}_seed*.jsonl" --out "$OUT/div"
done

echo "===== C1C3.t4 emitter set"
S="mc.humanfact=$INV_GE:$(roster "$G1/val" "gemma+ent_humanfact" "_div1.jsonl")"
S="$S mc.humanfact.distilled=$INV_DI:$(roster "$G1/val" "distilled_humanfact" "_div1.jsonl")"
S="$S mc.judgefact=$INV_GE:$(roster "$G1/val" "gemma+ent_xenc" "_div1.jsonl")"
S="$S mc.judgefact.distilled=$INV_DI:$(roster "$G1/val" "distilled_xenc" "_div1.jsonl")"
CTL="--control mc.judgefact --control mc.judgefact.distilled"
for a in humanfactshuffled:mc.humanfact.shuffled humanfactempty:mc.humanfact.empty \
         xencshuffled:mc.judgefact.shuffled xencempty:mc.judgefact.empty; do
  src="${a%%:*}"; nm="${a##*:}"
  S="$S ${nm}=$INV_GE:$(roster "$R3/val" "gemma+ent_${src}" "_div1.jsonl")"
  S="$S ${nm}.distilled=$INV_DI:$(roster "$R3/val" "distilled_${src}" "_div1.jsonl")"
  CTL="$CTL --control ${nm} --control ${nm}.distilled"
done
CMP=""
for fam in nli_summac_zs nli_nli lex_token_recall lex_rougeL_recall emb_minilm_maxcos; do
  S="$S zs.${fam}=$INV_GE:$(need "$R6/val/gemma+ent_${fam}_div1.jsonl")"
  S="$S zs.distilled.${fam}=$INV_DI:$(need "$R6/val/distilled_${fam}_div1.jsonl")"
  CMP="$CMP --comparator zs.${fam} --comparator zs.distilled.${fam}"
done
for n in fuse.mc_summac.distilled fuse.mc_nli.distilled fuse.mc_lex.distilled \
         fuse.mcshuf_summac.distilled fuse.mcempty_summac.distilled; do
  S="$S ${n}=$INV_DI:$(roster "$OUT/div" "$n" "_div1.jsonl")"; CTL="$CTL --control ${n}"
done
for n in fuse.mc_summac fuse.mcshuf_summac fuse.mcempty_summac; do
  S="$S ${n}=$INV_GE:$(roster "$OUT/div" "$n" "_div1.jsonl")"; CTL="$CTL --control ${n}"
done
B1DIV="$MCS/b1/test/div"
if [ -d "$B1DIV" ]; then
  for f in "$B1DIV"/gemma+ent_*_div1.jsonl; do [ -e "$f" ] || continue; b=$(basename "$f" _div1.jsonl); \
    S="$S zs.${b#gemma+ent_}=$INV_GE:$f"; CMP="$CMP --comparator zs.${b#gemma+ent_}"; done
  for f in "$B1DIV"/distilled_*_div1.jsonl;  do [ -e "$f" ] || continue; b=$(basename "$f" _div1.jsonl); \
    S="$S zs.distilled.${b#distilled_}=$INV_DI:$f"; CMP="$CMP --comparator zs.distilled.${b#distilled_}"; done
  echo "[c1c3-test] NOTE: block-B1 TEST verifier files found under $B1DIV and entered as comparators"
else
  echo "[c1c3-test] NOTE: no block-B1 TEST verifier files under $B1DIV -- MiniCheck and AlignScore are ABSENT"
  echo "[c1c3-test]       from this C1 comparator set. The hooks are in place; rerun this job when B1 lands."
fi
echo "===== C1C3.t5 emitters:"; for s in $S; do echo "  ${s%%=*}"; done

echo "===== C1C3.t6 evaluation (--k 10 alone, so --dump-emitted fires at k=10; explicit comma manifests only)"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU" --k 10 --n-boot 2000 \
  --primary "mc.humanfact" --out "$OUT/eval_test" \
  --dump-emitted "$OUT/emitted_test_k10.jsonl" --dump-per-pair "$OUT/per_pair_test_k10.jsonl" \
  --systems $S 2>&1 | grep -E "\[e2"

echo "===== C1C3.t7 budget-coverage cross-check (recall IS computable here; k is the coverage knob)"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU" --k 2 4 6 8 10 --n-boot 2000 \
  --primary "mc.humanfact.distilled" --out "$OUT/eval_test_kcurve" --systems \
  "mc.humanfact=$INV_GE:$(roster "$G1/val" "gemma+ent_humanfact" "_div1.jsonl")" \
  "mc.humanfact.distilled=$INV_DI:$(roster "$G1/val" "distilled_humanfact" "_div1.jsonl")" \
  "zs.distilled.nli_summac_zs=$INV_DI:$R6/val/distilled_nli_summac_zs_div1.jsonl" \
  "zs.nli_summac_zs=$INV_GE:$R6/val/gemma+ent_nli_summac_zs_div1.jsonl" \
  "fuse.mc_summac.distilled=$INV_DI:$(roster "$OUT/div" "fuse.mc_summac.distilled" "_div1.jsonl")" \
  "fuse.mc_summac=$INV_GE:$(roster "$OUT/div" "fuse.mc_summac" "_div1.jsonl")" 2>&1 | grep -E "\[e2"

echo "===== C1C3.t8 risk-coverage estimator (C1) and complementarity (C3), document-clustered"
"$PLAN_PY" "$SCODE/c1c3_riskcov.py" --emitted "$OUT/emitted_test_k10.jsonl" --acu "$ACU" \
  --out "$OUT/analysis" --n-boot 2000 --k-label 10 --bar-floor 0.20 \
  --label "sealed TEST, PREREG Wave C / C1 (H-SEL) and C3 (H-COMP)" \
  --primary mc.humanfact --primary mc.humanfact.distilled $CMP $CTL \
  --fusion "fuse.mc_summac.distilled=mc.humanfact.distilled+zs.distilled.nli_summac_zs" \
  --fusion "fuse.mc_nli.distilled=mc.humanfact.distilled+zs.distilled.nli_nli" \
  --fusion "fuse.mc_lex.distilled=mc.humanfact.distilled+zs.distilled.lex_token_recall" \
  --fusion "fuse.mcshuf_summac.distilled=mc.humanfact.shuffled.distilled+zs.distilled.nli_summac_zs" \
  --fusion "fuse.mcempty_summac.distilled=mc.humanfact.empty.distilled+zs.distilled.nli_summac_zs" \
  --fusion "fuse.mc_summac=mc.humanfact+zs.nli_summac_zs" \
  --fusion "fuse.mcshuf_summac=mc.humanfact.shuffled+zs.nli_summac_zs" \
  --fusion "fuse.mcempty_summac=mc.humanfact.empty+zs.nli_summac_zs" \
  --eval-json "$OUT/eval_test/e2_emitted.json" "$OUT/eval_test_kcurve/e2_emitted.json" 2>&1 | grep -E "\[c1|\[c3"

(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[c1c3-test] done -> $OUT/analysis/c1c3_result.json"
