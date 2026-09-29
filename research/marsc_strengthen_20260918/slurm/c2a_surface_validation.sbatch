#!/bin/bash
# MARS-C strengthen C2 / H-SURF (a) -- fit the surface model on VALIDATION, publish the diagnostic R^2 table,
# and re-evaluate every validation arm on the residualised score through the identical div1 rule and evaluator.
#
# Registered in research/marsc_strengthen_20260918/PREREG.md, Wave C / C2.
#
# The reviewer question the paper currently cannot answer: has the verifier learned that LONG units, or units
# LATE in the source, or units LEXICALLY DISTANT from the summary, are simply more often omitted -- rather than
# learning coverage?  This job regresses each arm's per-unit omission score on those three surface cues, prints
# R^2 and the coefficients per family, subtracts the fitted surface prediction, and runs the residualised score
# back through mc_diversity.py -> mc_e2_eval.py unchanged.
#
# Identification contract: the model is FITTED HERE, on validation, and the fitted file is what c2b applies to
# the sealed TEST split and to UniSumEval.  Nothing of an evaluation split enters the transform.
# Two residual variants are produced in the same pass:
#   +res       full three-regressor model (the registered variant the C2 bar is stated on)
#   +resblind  candidate-BLIND regressors only (position + length), which removes omission PRIORS and leaves
#              whatever conditioning on the actual summary the verifier has
#SBATCH --job-name=mars-c2a-surface-val
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
OUT="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}/c2c4"
JOB="$OUT/c2a_validation"
mkdir -p "$OUT/models" "$OUT/res/validation" "$OUT/resblind/validation" \
         "$OUT/div/validation_res" "$OUT/div/validation_resblind" "$JOB"
plan_receipt "$JOB"

ACU_V="$G/b18/acu_units_validation.jsonl"
INV_GE="$M3/m32/units_gemma+ent_validation.jsonl"
INV_DI="$M3/m32/units_distilled_validation.jsonl"
INV_SP="$M3/m32/units_spacy_validation.jsonl"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"

# explicit per-seed manifest; exits 4 on a missing seed file (mc_r12_precision_endpoint.sbatch's helper)
ros () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; \
  [ -f "$f" ] || { echo "[c2a] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
C2 () { "$PLAN_PY" "$SCODE/c2_surface_control.py" "$@"; }
DIV () { "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "$1" --scores-glob "$2" --out "$3"; }
EV () { "$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 "$@"; }

# ---- frozen inputs: the five validation arms, raw scores and their published div1 outputs
RAW_HF=$(ros "$MC/scores"              "units_gemma+ent_mc-A-lam1-crossenc-inv" ".jsonl")
RAW_JF=$(ros "$M3/scores"              "units_gemma+ent_xenc"                   ".jsonl")
RAW_HFD=$(ros "$MC/e2imp_all/scores"   "distilled_humanfact"                    ".jsonl")
RAW_JFD=$(ros "$M3/scores"             "units_distilled_xenc"                   ".jsonl")
RAW_SP=$(ros "$M3/scores"              "units_spacy_tagger"                     ".jsonl")
DIV_HF=$(ros "$MC/e2div/val"           "units_gemma+ent_mc-A-lam1-crossenc-inv" "_div1.jsonl")
DIV_JF=$(ros "$MC/e2div_full/val"      "units_gemma+ent_xenc"                   "_div1.jsonl")
DIV_HFD=$(ros "$MC/e2div/val"          "distilled_humanfact"                    "_div1.jsonl")
DIV_JFD=$(ros "$MC/e2div/val"          "units_distilled_xenc"                   "_div1.jsonl")
DIV_SP=$(ros "$MC/e2div_full/val"      "units_spacy_tagger"                     "_div1.jsonl")

echo "===== C2a.1 fit the surface model on VALIDATION (one OLS per arm per seed)"
C2 --mode fit --arm humanfact           --units "gemma+ent=$INV_GE" --scores "$RAW_HF"  --require-split validation --model-out "$OUT/models/surface_humanfact.json"
C2 --mode fit --arm judgefact           --units "gemma+ent=$INV_GE" --scores "$RAW_JF"  --require-split validation --model-out "$OUT/models/surface_judgefact.json"
C2 --mode fit --arm humanfact.distilled --units "distilled=$INV_DI" --scores "$RAW_HFD" --require-split validation --model-out "$OUT/models/surface_humanfact.distilled.json"
C2 --mode fit --arm judgefact.distilled --units "distilled=$INV_DI" --scores "$RAW_JFD" --require-split validation --model-out "$OUT/models/surface_judgefact.distilled.json"
C2 --mode fit --arm tagger.spacy        --units "spacy=$INV_SP"     --scores "$RAW_SP"  --require-split validation --model-out "$OUT/models/surface_tagger.spacy.json"

echo "===== C2a.2 the diagnostic table the paper prints"
C2 --mode table --out "$OUT/models/c2_surface_table.json" --models \
  "$OUT/models/surface_humanfact.json" "$OUT/models/surface_judgefact.json" \
  "$OUT/models/surface_humanfact.distilled.json" "$OUT/models/surface_judgefact.distilled.json" \
  "$OUT/models/surface_tagger.spacy.json"

echo "===== C2a.3 residualise every validation arm (both variants) and re-apply the div1 rule unchanged"
for V in res resblind; do
  C2 --mode apply --variant "$V" --arm humanfact           --units "gemma+ent=$INV_GE" --scores "$RAW_HF"  --model "$OUT/models/surface_humanfact.json"           --out "$OUT/$V/validation" --manifest-out "$JOB/apply_${V}_humanfact.json"
  C2 --mode apply --variant "$V" --arm judgefact           --units "gemma+ent=$INV_GE" --scores "$RAW_JF"  --model "$OUT/models/surface_judgefact.json"           --out "$OUT/$V/validation" --manifest-out "$JOB/apply_${V}_judgefact.json"
  C2 --mode apply --variant "$V" --arm humanfact.distilled --units "distilled=$INV_DI" --scores "$RAW_HFD" --model "$OUT/models/surface_humanfact.distilled.json" --out "$OUT/$V/validation" --manifest-out "$JOB/apply_${V}_humanfact.distilled.json"
  C2 --mode apply --variant "$V" --arm judgefact.distilled --units "distilled=$INV_DI" --scores "$RAW_JFD" --model "$OUT/models/surface_judgefact.distilled.json" --out "$OUT/$V/validation" --manifest-out "$JOB/apply_${V}_judgefact.distilled.json"
  C2 --mode apply --variant "$V" --arm tagger.spacy        --units "spacy=$INV_SP"     --scores "$RAW_SP"  --model "$OUT/models/surface_tagger.spacy.json"        --out "$OUT/$V/validation" --manifest-out "$JOB/apply_${V}_tagger.spacy.json"
  DIV "gemma+ent=$INV_GE" "$OUT/$V/validation/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*.jsonl" "$OUT/div/validation_$V"
  DIV "gemma+ent=$INV_GE" "$OUT/$V/validation/units_gemma+ent_xenc_seed*.jsonl"                   "$OUT/div/validation_$V"
  DIV "distilled=$INV_DI" "$OUT/$V/validation/distilled_humanfact_seed*.jsonl"                    "$OUT/div/validation_$V"
  DIV "distilled=$INV_DI" "$OUT/$V/validation/units_distilled_xenc_seed*.jsonl"                   "$OUT/div/validation_$V"
  DIV "spacy=$INV_SP"     "$OUT/$V/validation/units_spacy_tagger_seed*.jsonl"                     "$OUT/div/validation_$V"
done

RS="$OUT/div/validation_res"; RB="$OUT/div/validation_resblind"
RES_HF=$(ros  "$RS" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_div1.jsonl"); RSB_HF=$(ros  "$RB" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_div1.jsonl")
RES_JF=$(ros  "$RS" "units_gemma+ent_xenc"                   "_div1.jsonl"); RSB_JF=$(ros  "$RB" "units_gemma+ent_xenc"                   "_div1.jsonl")
RES_HFD=$(ros "$RS" "distilled_humanfact"                    "_div1.jsonl"); RSB_HFD=$(ros "$RB" "distilled_humanfact"                    "_div1.jsonl")
RES_JFD=$(ros "$RS" "units_distilled_xenc"                   "_div1.jsonl"); RSB_JFD=$(ros "$RB" "units_distilled_xenc"                   "_div1.jsonl")
RES_SP=$(ros  "$RS" "units_spacy_tagger"                     "_div1.jsonl"); RSB_SP=$(ros  "$RB" "units_spacy_tagger"                     "_div1.jsonl")

SPECS="humanfact+div1=$INV_GE:$DIV_HF \
humanfact.res+div1=$INV_GE:$RES_HF \
humanfact.resblind+div1=$INV_GE:$RSB_HF \
judgefact+div1=$INV_GE:$DIV_JF \
judgefact.res+div1=$INV_GE:$RES_JF \
judgefact.resblind+div1=$INV_GE:$RSB_JF \
humanfact.distilled+div1=$INV_DI:$DIV_HFD \
humanfact.distilled.res+div1=$INV_DI:$RES_HFD \
humanfact.distilled.resblind+div1=$INV_DI:$RSB_HFD \
judgefact.distilled+div1=$INV_DI:$DIV_JFD \
judgefact.distilled.res+div1=$INV_DI:$RES_JFD \
judgefact.distilled.resblind+div1=$INV_DI:$RSB_JFD \
tagger.spacy+div1=$INV_SP:$DIV_SP \
tagger.spacy.res+div1=$INV_SP:$RES_SP \
tagger.spacy.resblind+div1=$INV_SP:$RSB_SP"

echo "===== C2a.4 validation: every raw arm beside its residualised twin, primary = the RAW MARS-C arm"
EV --acu "$ACU_V" --k 10 --n-boot 2000 --robust-tests --n-perm 10000 --primary "humanfact+div1" \
   --out "$JOB/eval_raw_vs_res" --systems $SPECS 2>&1 | grep -E "^\[e2 (k|robust)"

echo "===== C2a.5 validation: the residualised field on its own, primary = the RESIDUALISED MARS-C arm"
EV --acu "$ACU_V" --k 10 --n-boot 2000 --robust-tests --n-perm 10000 --primary "humanfact.res+div1" \
   --out "$JOB/eval_res_primary" --systems \
  "humanfact.res+div1=$INV_GE:$RES_HF" "judgefact.res+div1=$INV_GE:$RES_JF" \
  "humanfact.distilled.res+div1=$INV_DI:$RES_HFD" "judgefact.distilled.res+div1=$INV_DI:$RES_JFD" \
  "tagger.spacy.res+div1=$INV_SP:$RES_SP" \
  "humanfact.resblind+div1=$INV_GE:$RSB_HF" "judgefact.resblind+div1=$INV_GE:$RSB_JF" \
  "humanfact.distilled.resblind+div1=$INV_DI:$RSB_HFD" "judgefact.distilled.resblind+div1=$INV_DI:$RSB_JFD" \
  "tagger.spacy.resblind+div1=$INV_SP:$RSB_SP" 2>&1 | grep -E "^\[e2 (k|robust)"

(cd "$OUT" && find models c2a_validation -type f -name "*.json" | sort | xargs sha256sum > c2a_validation/SHA256SUMS)
plan_done "$JOB"
echo "[c2a] done -> $JOB"
