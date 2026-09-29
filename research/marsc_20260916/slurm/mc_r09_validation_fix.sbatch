#!/bin/bash
# MARS-C R09 -- REPAIR of the validation-split comparisons.
#
# The glob audit (mc_glob_audit.py, R08) found a SECOND instance of the consensus defect that the external
# review did not: on validation the diversity outputs of the three verifier seeds live in the same directory as
# the diversity outputs of the REJECTED learned-importance policy (`*_seed<N>_imp0.2_div1.jsonl`), and
# `*_seed*_div1.jsonl` matches both. Every validation "consensus" cell in E2/G3 therefore averaged six files:
# three rule outputs and three importance-policy outputs. This job re-runs the whole validation comparison from
# explicit three-file manifests under --strict-seeds, which refuses a roster whose seeds are not distinct.
#SBATCH --job-name=mc-r09-valfix
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
DIV="$MC/e2div"; DF="$MC/e2div_full"; IMP2="$MC/e2imp_all"; CTL="$MC/e2ctl"; G3="$MC/g3"
OUT="$MC/r09_validation"; mkdir -p "$OUT/val"; plan_receipt "$OUT"
INV_GE="$M3/m32/units_gemma+ent_validation.jsonl"; INV_DI="$M3/m32/units_distilled_validation.jsonl"
INV_SP="$M3/m32/units_spacy_validation.jsonl"; INV_UN="$G3/inv/units_union_validation.jsonl"
INV_M2="$G3/inv/units_mars2_validation.jsonl"; ACU_V="$G/b18/acu_units_validation.jsonl"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || { echo "[r09] MISSING $f" >&2; return 1; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
div () { "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "$1" --scores-glob "$2" --out "$OUT/val"; }

echo "===== R09.0 rebuild the union diversity outputs (the G3 union glob matched nothing and was silently dropped)"
for S in $SEEDS; do
  div "union=$INV_UN" "$G3/inv/union_gemma+ent_mc-A-lam1-crossenc-inv_seed${S}.jsonl"
  div "union=$INV_UN" "$G3/inv/union_gemma+ent_xenc_seed${S}.jsonl"
done

HF_GE=$(roster "$DIV/val" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_div1.jsonl")
HF_DI=$(roster "$DIV/val" "distilled_humanfact" "_div1.jsonl")
JF_DI=$(roster "$DIV/val" "units_distilled_xenc" "_div1.jsonl")
JF_GE=$(roster "$DF/val" "units_gemma+ent_xenc" "_div1.jsonl")
UN_HF=$(roster "$OUT/val" "union_gemma+ent_mc-A-lam1-crossenc-inv" "_div1.jsonl")
UN_JF=$(roster "$OUT/val" "union_gemma+ent_xenc" "_div1.jsonl")
CAP2=$(roster "$G3/val" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_div2.jsonl")
IMP_GE=$(roster "$IMP2/val" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_imp0.2.jsonl")
IMP_DI=$(roster "$IMP2/val" "distilled_humanfact" "_imp0.2.jsonl")
LEAD_GE=$(roster "$CTL/val_lead" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_imp0.2.jsonl")
RAND_GE=$(roster "$CTL/val_rand" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_imp0.795.jsonl")
RAW_GE=$(roster "$MC/scores" "units_gemma+ent_mc-A-lam1-crossenc-inv" ".jsonl")
RAW_DI=$(roster "$IMP2/scores" "distilled_humanfact" ".jsonl")
RAW_JFDI=$(roster "$M3/scores" "units_distilled_xenc" ".jsonl")
TAG_SP=$(roster "$DF/val" "units_spacy_tagger" "_div1.jsonl")

echo "===== R09.1 the full validation comparison from explicit three-seed manifests"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU_V" --k 5 10 20 --n-boot 500 \
  --primary "humanfact+div1" --robust-tests --n-perm 10000 --out "$OUT/eval_validation" --systems \
  "humanfact+div1=$INV_GE:$HF_GE" \
  "humanfact.distilled+div1=$INV_DI:$HF_DI" \
  "judgefact.distilled+div1=$INV_DI:$JF_DI" \
  "judgefact+div1=$INV_GE:$JF_GE" \
  "union_humanfact+div1=$INV_UN:$UN_HF" \
  "union_judgefact+div1=$INV_UN:$UN_JF" \
  "ensemble+div1=$INV_GE:$G3/val/gemma+ent_ens_seedens_div1.jsonl" \
  "ensemble.distilled+div1=$INV_DI:$G3/val/distilled_ens_seedens_div1.jsonl" \
  "humanfact+div2=$INV_GE:$CAP2" \
  "humanfact+imp=$INV_GE:$IMP_GE" \
  "humanfact+leadpos=$INV_GE:$LEAD_GE" \
  "humanfact+random=$INV_GE:$RAND_GE" \
  "humanfact.distilled+imp=$INV_DI:$IMP_DI" \
  "humanfact=$INV_GE:$RAW_GE" \
  "humanfact.distilled=$INV_DI:$RAW_DI" \
  "judgefact.distilled=$INV_DI:$RAW_JFDI" \
  "tagger.spacy+div1=$INV_SP:$TAG_SP" \
  "mars2+div1=$INV_M2:$G3/val/mars2_mars2_seed0_div1.jsonl" \
  "mars2=$INV_M2:$G3/inv/mars2_mars2_seed0.jsonl" 2>&1 | grep -E "\[e2"
(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[r09] done -> $OUT"
