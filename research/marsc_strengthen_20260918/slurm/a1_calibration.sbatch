#!/bin/bash
# MARS-C A1 -- the calibration contract. Registered in research/marsc_strengthen_20260918/PREREG.md, Wave A / A1.
# This is the strengthening campaign's ENTRY GATE: no other block's number is citable until it closes.
#
# Defect. mc_train.py fits a per-seed Platt map on its calibration documents and applies it to every score file
# it writes; mc_score_units.py emits a bare sigmoid. The contract is mixed WITHIN validation -- the primary
# `humanfact` cell on gemma+ent is Platt-shrunk while `humanfact.distilled` and every `judgefact` cell are raw.
#
# A1a  declares ONE contract (raw sigmoid everywhere) and re-derives every validation cell under it. The Platt
#      map is inverted analytically from {slope, intercept} in each arm's result.json; the inversion is GATED
#      against the raw-logit companions mc_train wrote beside the calibrated validation-ACU files (27 files,
#      3 arms x seeds), and this job exits 5 without producing a single downstream number if it does not
#      reproduce them to 1e-6. Both contracts are rebuilt through the SAME code path in this job's own output
#      tree, so the reported delta is attributable to the contract alone and to nothing else.
# A1b  the invariance result: how many of the 1466 validation pairs change their top-10 SET under the map,
#      per seed and per aggregation regime. Rank-only rule + strictly monotone map => single-seed is exactly
#      invariant; the exposure lives in seed AVERAGING.
# A1c  the label-efficiency curve (0.4085/0.4023/0.3953/0.3982) re-derived raw. Its four points were fitted at
#      DIFFERENT calibration-set sizes (n_cells 2772/5385/7225/7225, slopes 0.558-0.841), so its reported
#      spread of 0.0132 is the same order as the calibration perturbation. Underwrites R05.
# A1d  the R04 matched label-source AUROC recomputed by averaging RAW LOGITS. mc_matched_auroc.py seed-averages
#      CALIBRATED probabilities, and the two arms' maps were fitted at different prevalences with
#      opposite-signed intercepts (human rate 0.6094, icpt +0.159/+0.355/+0.365; judge 0.3219, icpt
#      -0.444/-0.351/-0.170), so the averaging step is not label-source-neutral.
# A1e  leadunion on validation. Its primary sort key is per-seed sentence-leader COUNTS, exactly
#      calibration-invariant; it already reproduces the consensus headline to 3e-4 on TEST and on UniSumEval.
#
# No bar: A1 is a repair. The corrected number is the number, whichever way it moves.
# Validation only; no sealed split is read, so no MC_TEST_REREAD gate applies. CPU only.
#SBATCH --job-name=mars-a1-calib
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
A1="$SCODE/a1_calibration.py"

SOUT="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}/a1"
CAL="$SOUT/cal"; RAW="$SOUT/raw"
mkdir -p "$SOUT/records" "$CAL/inv" "$CAL/val" "$CAL/val_lead" "$CAL/val_rand" "$CAL/agg" "$CAL/t07" "$CAL/t07val" \
         "$RAW/inv" "$RAW/val" "$RAW/val_lead" "$RAW/val_rand" "$RAW/agg" "$RAW/t07" "$RAW/t07val"
plan_receipt "$SOUT"

SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
NBOOT="${A1_NBOOT:-1000}"
INV_GE="$M3/m32/units_gemma+ent_validation.jsonl"
INV_DI="$M3/m32/units_distilled_validation.jsonl"
INV_SP="$M3/m32/units_spacy_validation.jsonl"
INV_M2="$MC/g3/inv/units_mars2_validation.jsonl"
ACU_V="$G/b18/acu_units_validation.jsonl"
DIV="$MC/e2div"; DF="$MC/e2div_full"; IMP2="$MC/e2imp_all"; CTL="$MC/e2ctl"; G3="$MC/g3"; R4="$MC/r04_matched"; T7="$MC/t07_curve"
HF_GE_SRC="$MC/scores"                 # calibrated primary inventory scores (written by mc_train.py)
HF_DI="$IMP2/scores/distilled_humanfact_seed*.jsonl"   # raw (mc_score_units.py)
JF_GE="$M3/scores/units_gemma+ent_xenc_seed*.jsonl"    # raw
JF_DI="$M3/scores/units_distilled_xenc_seed*.jsonl"    # raw

# explicit three-seed manifest; never a glob into mc_e2_eval (TRAP 1)
roster () { local dir="$1" pre="$2" suf="$3" out="" s f
  for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || { echo "[a1] MISSING $f" >&2; return 1; }
    out="${out:+$out,}$f"; done; printf '%s' "$out"; }
nglob () { local n; n=$(ls $1 2>/dev/null | wc -l); [ "$n" = "$2" ] || { echo "[a1] expected $2 files for $1, found $n" >&2; exit 4; }; }

echo "===== A1.0  GATE: the analytic Platt inversion must reproduce the stored raw logits"
TRIPLES=""
for S in $SEEDS; do
  TRIPLES="$TRIPLES primary-inv_seed$S:$MC/scores/acu_mc-A-lam1-crossenc-inv_seed$S.jsonl:$MC/scores/acu_mc-A-lam1-crossenc-inv_seed${S}_logits.jsonl:$MC/e1/mc-A-lam1-crossenc-inv_seed$S/result.json"
done
for FR in 10 25 50 100; do for S in $SEEDS; do
  TRIPLES="$TRIPLES t07-frac${FR}_seed$S:$T7/scores/acu_t07-frac${FR}_seed$S.jsonl:$T7/scores/acu_t07-frac${FR}_seed${S}_logits.jsonl:$T7/t07-frac${FR}_seed$S/result.json"
done; done
for ARM in mc-A-lam1-crossenc-r04human mc-A-lam1-crossenc-r04judge mc-R-lam1-crossenc-r04human mc-R-lam1-crossenc-r04judge; do
  for S in $SEEDS; do
    TRIPLES="$TRIPLES ${ARM}_seed$S:$R4/scores/acu_${ARM}_seed$S.jsonl:$R4/scores/acu_${ARM}_seed${S}_logits.jsonl:$R4/models/${ARM}_seed$S/result.json"
  done
done
"$PLAN_PY" "$A1" verify --tol 1e-6 --out "$SOUT/a1a_inversion.json" --triple $TRIPLES

echo "===== A1.1  build both contracts of the primary arm through the SAME code path"
for S in $SEEDS; do
  cp -f "$HF_GE_SRC/units_gemma+ent_mc-A-lam1-crossenc-inv_seed$S.jsonl" "$CAL/inv/"
  "$PLAN_PY" "$A1" invert \
    --in "$HF_GE_SRC/units_gemma+ent_mc-A-lam1-crossenc-inv_seed$S.jsonl" \
    --result-json "$MC/e1/mc-A-lam1-crossenc-inv_seed$S/result.json" \
    --out "$RAW/inv/units_gemma+ent_mc-A-lam1-crossenc-inv_seed$S.jsonl" \
    --record "$SOUT/records/invert_primary_seed$S.json"
done
nglob "$CAL/inv/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*.jsonl" 3
nglob "$RAW/inv/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*.jsonl" 3

build () {   # $1 = contract root ($CAL or $RAW)
  local D="$1"
  local HFG="$D/inv/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*.jsonl"
  echo "----- build $D"
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "gemma+ent=$INV_GE" --scores-glob "$HFG" --out "$D/val"
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "gemma+ent=$INV_GE" --scores-glob "$HFG" --out "$D/val" --cap 2
  # single-model regime: average the seeds, THEN apply the rule
  "$PLAN_PY" "$MCCODE/mc_variants.py" seedavg --units "gemma+ent=$INV_GE" --scores "$HFG" --tag humanfact --out "$D/agg"
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "gemma+ent=$INV_GE" --scores-glob "$D/agg/gemma+ent_humanfact_seedavg.jsonl" --out "$D/val"
  # leadunion: sentence-leader counts first, mean probability only as tie-break (A1e)
  "$PLAN_PY" "$MCCODE/mc_aggregate.py" leadunion --units "gemma+ent=$INV_GE" --scores "$HFG" --tag humanfact --out "$D/agg"
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "gemma+ent=$INV_GE" --scores-glob "$D/agg/gemma+ent_humanfact_seedlead.jsonl" --out "$D/val"
  # verifier ensemble against the judge-label arm (that arm is raw in both contracts)
  "$PLAN_PY" "$MCCODE/mc_variants.py" ensemble --units "gemma+ent=$INV_GE" --scores-a "$HFG" --scores-b "$JF_GE" --tag ens --out "$D/agg"
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "gemma+ent=$INV_GE" --scores-glob "$D/agg/gemma+ent_ens_seedens.jsonl" --out "$D/val"
  # union inventory, both verifier arms, rebuilt here so the units file and the score files always agree
  "$PLAN_PY" "$MCCODE/mc_variants.py" union --units "gemma+ent=$INV_GE" "distilled=$INV_DI" \
    --scores-a "$HFG" "$JF_GE" --scores-b "$HF_DI" "$JF_DI" --name union --split validation --out "$D/agg"
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "union=$D/agg/units_union_validation.jsonl" --scores-glob "$D/agg/union_gemma+ent_mc-A-lam1-crossenc-inv_seed*.jsonl" --out "$D/val"
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "union=$D/agg/units_union_validation.jsonl" --scores-glob "$D/agg/union_gemma+ent_xenc_seed*.jsonl" --out "$D/val"
  # the learned importance policy and its two registered controls, on the same omission scores
  "$PLAN_PY" "$MCCODE/mc_importance.py" combine --omission-glob "$HFG" --importance-glob "$IMP2/scores/gemma+ent_importance_all_seed*.jsonl" --tau 0.2   --out "$D/val"
  "$PLAN_PY" "$MCCODE/mc_importance.py" combine --omission-glob "$HFG" --importance-glob "$CTL/scores/gemma+ent_leadpos.jsonl" --tau 0.2   --out "$D/val_lead"
  "$PLAN_PY" "$MCCODE/mc_importance.py" combine --omission-glob "$HFG" --importance-glob "$CTL/scores/gemma+ent_random.jsonl"  --tau 0.795 --out "$D/val_rand"
}
build "$CAL"
build "$RAW"

echo "===== A1.2  every validation cell, under both contracts, from explicit three-seed manifests"
HF_DI_DIV=$(roster "$DIV/val" "distilled_humanfact" "_div1.jsonl")           || exit 4
HF_DI_RAW=$(roster "$IMP2/scores" "distilled_humanfact" ".jsonl")            || exit 4
HF_DI_IMP=$(roster "$IMP2/val" "distilled_humanfact" "_imp0.2.jsonl")        || exit 4
JF_GE_DIV=$(roster "$DF/val" "units_gemma+ent_xenc" "_div1.jsonl")           || exit 4
JF_DI_DIV=$(roster "$DIV/val" "units_distilled_xenc" "_div1.jsonl")          || exit 4
JF_DI_RAW=$(roster "$M3/scores" "units_distilled_xenc" ".jsonl")             || exit 4
TAG_SP=$(roster "$DF/val" "units_spacy_tagger" "_div1.jsonl")                || exit 4

evalrun () {   # $1 = contract root  $2 = eval out dir
  local D="$1" O="$2" UN="$1/agg/units_union_validation.jsonl"
  local HF_RAW HF_D1 HF_D2 HF_IMP HF_LEAD HF_RAND UN_HF UN_JF
  HF_RAW=$(roster  "$D/inv"      "units_gemma+ent_mc-A-lam1-crossenc-inv" ".jsonl")       || exit 4
  HF_D1=$(roster   "$D/val"      "units_gemma+ent_mc-A-lam1-crossenc-inv" "_div1.jsonl")  || exit 4
  HF_D2=$(roster   "$D/val"      "units_gemma+ent_mc-A-lam1-crossenc-inv" "_div2.jsonl")  || exit 4
  HF_IMP=$(roster  "$D/val"      "units_gemma+ent_mc-A-lam1-crossenc-inv" "_imp0.2.jsonl")   || exit 4
  HF_LEAD=$(roster "$D/val_lead" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_imp0.2.jsonl")   || exit 4
  HF_RAND=$(roster "$D/val_rand" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_imp0.795.jsonl") || exit 4
  UN_HF=$(roster   "$D/val"      "union_gemma+ent_mc-A-lam1-crossenc-inv" "_div1.jsonl")  || exit 4
  UN_JF=$(roster   "$D/val"      "union_gemma+ent_xenc" "_div1.jsonl")                    || exit 4
  "$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU_V" --k 5 10 20 \
    --n-boot "$NBOOT" --primary "humanfact+div1" --out "$O" --systems \
    "humanfact+div1=$INV_GE:$HF_D1" \
    "humanfact+div2=$INV_GE:$HF_D2" \
    "humanfact=$INV_GE:$HF_RAW" \
    "humanfact.single+div1=$INV_GE:$D/val/gemma+ent_humanfact_seedavg_div1.jsonl" \
    "humanfact.lead+div1=$INV_GE:$D/val/gemma+ent_humanfact_seedlead_div1.jsonl" \
    "humanfact+imp=$INV_GE:$HF_IMP" \
    "humanfact+leadpos=$INV_GE:$HF_LEAD" \
    "humanfact+random=$INV_GE:$HF_RAND" \
    "union_humanfact+div1=$UN:$UN_HF" \
    "union_judgefact+div1=$UN:$UN_JF" \
    "ensemble+div1=$INV_GE:$D/val/gemma+ent_ens_seedens_div1.jsonl" \
    "ensemble.distilled+div1=$INV_DI:$G3/val/distilled_ens_seedens_div1.jsonl" \
    "humanfact.distilled+div1=$INV_DI:$HF_DI_DIV" \
    "humanfact.distilled=$INV_DI:$HF_DI_RAW" \
    "humanfact.distilled+imp=$INV_DI:$HF_DI_IMP" \
    "judgefact+div1=$INV_GE:$JF_GE_DIV" \
    "judgefact.distilled+div1=$INV_DI:$JF_DI_DIV" \
    "judgefact.distilled=$INV_DI:$JF_DI_RAW" \
    "tagger.spacy+div1=$INV_SP:$TAG_SP" \
    "mars2+div1=$INV_M2:$G3/val/mars2_mars2_seed0_div1.jsonl" \
    "mars2=$INV_M2:$G3/inv/mars2_mars2_seed0.jsonl" 2>&1 | grep -E "^\[e2"
}
evalrun "$CAL" "$SOUT/eval_val_cal"
evalrun "$RAW" "$SOUT/eval_val_raw"
"$PLAN_PY" "$A1" collect --k 10 --baseline calibrated --registered "PREREG.md Wave A / A1a" \
  --eval "calibrated=$SOUT/eval_val_cal" "raw=$SOUT/eval_val_raw" --out "$SOUT/a1a.json"

echo "===== A1.3  the invariance measurement (A1b): pairs whose top-10 SET moves under the map"
CMP=""
for S in $SEEDS; do
  CMP="$CMP single_seed_${S}_noRule=$CAL/inv/units_gemma+ent_mc-A-lam1-crossenc-inv_seed$S.jsonl|$RAW/inv/units_gemma+ent_mc-A-lam1-crossenc-inv_seed$S.jsonl"
  CMP="$CMP single_seed_${S}_div1=$CAL/val/units_gemma+ent_mc-A-lam1-crossenc-inv_seed${S}_div1.jsonl|$RAW/val/units_gemma+ent_mc-A-lam1-crossenc-inv_seed${S}_div1.jsonl"
done
C_RAWINV=$(roster "$CAL/inv" "units_gemma+ent_mc-A-lam1-crossenc-inv" ".jsonl")      || exit 4
R_RAWINV=$(roster "$RAW/inv" "units_gemma+ent_mc-A-lam1-crossenc-inv" ".jsonl")      || exit 4
C_DIV1=$(roster   "$CAL/val" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_div1.jsonl") || exit 4
R_DIV1=$(roster   "$RAW/val" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_div1.jsonl") || exit 4
CMP="$CMP consensus_noRule=$C_RAWINV|$R_RAWINV"
CMP="$CMP consensus_div1=$C_DIV1|$R_DIV1"
CMP="$CMP single_model_noRule=$CAL/agg/gemma+ent_humanfact_seedavg.jsonl|$RAW/agg/gemma+ent_humanfact_seedavg.jsonl"
CMP="$CMP single_model_div1=$CAL/val/gemma+ent_humanfact_seedavg_div1.jsonl|$RAW/val/gemma+ent_humanfact_seedavg_div1.jsonl"
CMP="$CMP leadunion_noRule=$CAL/agg/gemma+ent_humanfact_seedlead.jsonl|$RAW/agg/gemma+ent_humanfact_seedlead.jsonl"
CMP="$CMP leadunion_div1=$CAL/val/gemma+ent_humanfact_seedlead_div1.jsonl|$RAW/val/gemma+ent_humanfact_seedlead_div1.jsonl"
"$PLAN_PY" "$A1" topk --units "gemma+ent=$INV_GE" --k 10 --out "$SOUT/a1b.json" --compare $CMP

echo "===== A1.4  the label-efficiency curve on the same contract (A1c)"
for FR in 10 25 50 100; do
  for S in $SEEDS; do
    cp -f "$T7/scores/gemma+ent_t07-frac${FR}_seed$S.jsonl" "$CAL/t07/"
    "$PLAN_PY" "$A1" invert \
      --in "$T7/scores/gemma+ent_t07-frac${FR}_seed$S.jsonl" \
      --result-json "$T7/t07-frac${FR}_seed$S/result.json" \
      --out "$RAW/t07/gemma+ent_t07-frac${FR}_seed$S.jsonl" \
      --record "$SOUT/records/invert_t07-frac${FR}_seed$S.json"
  done
  nglob "$CAL/t07/gemma+ent_t07-frac${FR}_seed*.jsonl" 3
  nglob "$RAW/t07/gemma+ent_t07-frac${FR}_seed*.jsonl" 3
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "gemma+ent=$INV_GE" --scores-glob "$CAL/t07/gemma+ent_t07-frac${FR}_seed*.jsonl" --out "$CAL/t07val"
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "gemma+ent=$INV_GE" --scores-glob "$RAW/t07/gemma+ent_t07-frac${FR}_seed*.jsonl" --out "$RAW/t07val"
done
curverun () {   # $1 = t07val dir  $2 = eval out dir
  local SPECS="" FR M
  for FR in 10 25 50 100; do
    M=$(roster "$1" "gemma+ent_t07-frac${FR}" "_div1.jsonl") || exit 4
    SPECS="$SPECS frac${FR}+div1=$INV_GE:$M"
  done
  "$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU_V" --k 5 10 \
    --n-boot "$NBOOT" --primary "frac100+div1" --out "$2" --systems $SPECS 2>&1 | grep -E "^\[e2"
}
curverun "$CAL/t07val" "$SOUT/eval_curve_cal"
curverun "$RAW/t07val" "$SOUT/eval_curve_raw"
"$PLAN_PY" "$A1" collect --k 10 --baseline calibrated --registered "PREREG.md Wave A / A1c" \
  --eval "calibrated=$SOUT/eval_curve_cal" "raw=$SOUT/eval_curve_raw" \
  --spread-over "frac10+div1" "frac25+div1" "frac50+div1" "frac100+div1" --out "$SOUT/a1c.json"

echo "===== A1.5  the R04 matched label-source AUROC from RAW LOGITS (A1d)"
ARMS=""
for ARM in mc-A-lam1-crossenc-r04human mc-A-lam1-crossenc-r04judge mc-R-lam1-crossenc-r04human mc-R-lam1-crossenc-r04judge; do
  CALF=$(roster "$R4/scores" "acu_${ARM}" ".jsonl")        || exit 4
  LOGF=$(roster "$R4/scores" "acu_${ARM}" "_logits.jsonl") || exit 4
  ARMS="$ARMS $ARM:$CALF:$LOGF"
done
"$PLAN_PY" "$A1" auroc --acu "$ACU_V" --primary mc-A-lam1-crossenc-r04human \
  --n-boot 2000 --seed 20260917 --out "$SOUT/a1d.json" --arm $ARMS

echo "===== A1.6  leadunion on validation, and the calibration bound it implies (A1e)"
"$PLAN_PY" "$A1" leadcheck --k 10 --split validation \
  --eval "calibrated=$SOUT/eval_val_cal" "raw=$SOUT/eval_val_raw" \
  --consensus "humanfact+div1" --lead "humanfact.lead+div1" --single "humanfact.single+div1" \
  --external "test:0.376536:0.376803" "unisum:0.347054:0.347304" --out "$SOUT/a1e.json"

(cd "$SOUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS)
plan_done "$SOUT"
echo "[a1] done -> $SOUT"
