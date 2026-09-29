#!/bin/bash
# MARS-C strengthen C2 / H-SURF (b) -- apply the VALIDATION-FITTED surface model, unchanged, to the sealed TEST
# split and to the untouched UniSumEval pool, and re-evaluate every arm on the residualised score.
#
# Registered in research/marsc_strengthen_20260918/PREREG.md, Wave C / C2.  Depends on c2a, which writes the
# fitted models; this job never fits anything -- `c2_surface_control.py --mode apply` cannot fit, and it refuses
# a model that has no entry for a score file's seed, so no moment of an evaluation split can enter the transform.
#
# The B0 standing rule of the campaign is honoured here: every residualised TEST arm is reported beside its
# shuffled-summary and no-summary controls, residualised with its own family's validation model, and beside
# omission precision.  No arm in this job is reported on recall@k alone.
#SBATCH --job-name=mars-c2b-surface-heldout
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[c2b] REFUSED -- sealed TEST re-read needs MC_TEST_REREAD=yes"; exit 3; }
OUT="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}/c2c4"
JOB="$OUT/c2b_heldout"
mkdir -p "$OUT/res/test" "$OUT/resblind/test" "$OUT/res/unisum" "$OUT/resblind/unisum" \
         "$OUT/div/test_res" "$OUT/div/test_resblind" "$OUT/div/unisum_res" "$OUT/div/unisum_resblind" "$JOB"
plan_receipt "$JOB"

M="$OUT/models"
for f in surface_humanfact.json surface_judgefact.json surface_humanfact.distilled.json surface_judgefact.distilled.json; do
  [ -f "$M/$f" ] || { echo "[c2b] MISSING fitted model $M/$f -- run c2a first" >&2; exit 4; }
done

G1="$MC/g1_test"; R3="$MC/r03_sumblind"; G7="$MC/g7_unisum"
ACU_T="$G/b18/acu_units_test.jsonl"; ACU_U="$G7/acu_units_unisum.jsonl"
INV_GE="$M3/m32/units_gemma+ent_test.jsonl"; INV_DI="$M3/m32/units_distilled_test.jsonl"
INV_U="$G7/units_gemma+ent_unisum.jsonl"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"

ros () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; \
  [ -f "$f" ] || { echo "[c2b] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
C2 () { "$PLAN_PY" "$SCODE/c2_surface_control.py" "$@"; }
DIV () { "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "$1" --scores-glob "$2" --out "$3"; }
EV () { "$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 "$@"; }

RAW_HF=$(ros  "$G1/scores" "gemma+ent_humanfact" ".jsonl"); DIV_HF=$(ros  "$G1/val" "gemma+ent_humanfact" "_div1.jsonl")
RAW_JF=$(ros  "$G1/scores" "gemma+ent_xenc"      ".jsonl"); DIV_JF=$(ros  "$G1/val" "gemma+ent_xenc"      "_div1.jsonl")
RAW_HFD=$(ros "$G1/scores" "distilled_humanfact" ".jsonl"); DIV_HFD=$(ros "$G1/val" "distilled_humanfact" "_div1.jsonl")
RAW_JFD=$(ros "$G1/scores" "distilled_xenc"      ".jsonl"); DIV_JFD=$(ros "$G1/val" "distilled_xenc"      "_div1.jsonl")
RAW_HFS=$(ros "$R3/scores" "gemma+ent_humanfactshuffled" ".jsonl"); DIV_HFS=$(ros "$R3/val" "gemma+ent_humanfactshuffled" "_div1.jsonl")
RAW_HFE=$(ros "$R3/scores" "gemma+ent_humanfactempty"    ".jsonl"); DIV_HFE=$(ros "$R3/val" "gemma+ent_humanfactempty"    "_div1.jsonl")
RAW_JFS=$(ros "$R3/scores" "gemma+ent_xencshuffled"      ".jsonl"); DIV_JFS=$(ros "$R3/val" "gemma+ent_xencshuffled"      "_div1.jsonl")
RAW_JFE=$(ros "$R3/scores" "gemma+ent_xencempty"         ".jsonl"); DIV_JFE=$(ros "$R3/val" "gemma+ent_xencempty"         "_div1.jsonl")
RAW_HFU=$(ros "$G7/scores" "gemma+ent_humanfact" ".jsonl"); DIV_HFU=$(ros "$G7/val" "gemma+ent_humanfact" "_div1.jsonl")
RAW_JFU=$(ros "$G7/scores" "gemma+ent_xenc"      ".jsonl"); DIV_JFU=$(ros "$G7/val" "gemma+ent_xenc"      "_div1.jsonl")

echo "===== C2b.1 apply the frozen validation surface models to TEST (both variants, real + B0 controls)"
for V in res resblind; do
  C2 --mode apply --variant "$V" --arm humanfact           --units "gemma+ent=$INV_GE" --scores "$RAW_HF"  --model "$M/surface_humanfact.json"           --out "$OUT/$V/test" --manifest-out "$JOB/apply_test_${V}_humanfact.json"
  C2 --mode apply --variant "$V" --arm judgefact           --units "gemma+ent=$INV_GE" --scores "$RAW_JF"  --model "$M/surface_judgefact.json"           --out "$OUT/$V/test" --manifest-out "$JOB/apply_test_${V}_judgefact.json"
  C2 --mode apply --variant "$V" --arm humanfact.distilled --units "distilled=$INV_DI" --scores "$RAW_HFD" --model "$M/surface_humanfact.distilled.json" --out "$OUT/$V/test" --manifest-out "$JOB/apply_test_${V}_humanfact.distilled.json"
  C2 --mode apply --variant "$V" --arm judgefact.distilled --units "distilled=$INV_DI" --scores "$RAW_JFD" --model "$M/surface_judgefact.distilled.json" --out "$OUT/$V/test" --manifest-out "$JOB/apply_test_${V}_judgefact.distilled.json"
  # B0 controls: residualised with THEIR OWN FAMILY's validation model -- the only model that exists for them,
  # declared here rather than fitted on the control's own scores, which would be fitting on the evaluation split.
  C2 --mode apply --variant "$V" --allow-arm-mismatch --arm humanfact.shuffled --units "gemma+ent=$INV_GE" --scores "$RAW_HFS" --model "$M/surface_humanfact.json" --out "$OUT/$V/test" --manifest-out "$JOB/apply_test_${V}_humanfact.shuffled.json"
  C2 --mode apply --variant "$V" --allow-arm-mismatch --arm humanfact.empty    --units "gemma+ent=$INV_GE" --scores "$RAW_HFE" --model "$M/surface_humanfact.json" --out "$OUT/$V/test" --manifest-out "$JOB/apply_test_${V}_humanfact.empty.json"
  C2 --mode apply --variant "$V" --allow-arm-mismatch --arm judgefact.shuffled --units "gemma+ent=$INV_GE" --scores "$RAW_JFS" --model "$M/surface_judgefact.json" --out "$OUT/$V/test" --manifest-out "$JOB/apply_test_${V}_judgefact.shuffled.json"
  C2 --mode apply --variant "$V" --allow-arm-mismatch --arm judgefact.empty    --units "gemma+ent=$INV_GE" --scores "$RAW_JFE" --model "$M/surface_judgefact.json" --out "$OUT/$V/test" --manifest-out "$JOB/apply_test_${V}_judgefact.empty.json"
  for P in gemma+ent_humanfact gemma+ent_xenc gemma+ent_humanfactshuffled gemma+ent_humanfactempty gemma+ent_xencshuffled gemma+ent_xencempty; do
    DIV "gemma+ent=$INV_GE" "$OUT/$V/test/${P}_seed*.jsonl" "$OUT/div/test_$V"
  done
  DIV "distilled=$INV_DI" "$OUT/$V/test/distilled_humanfact_seed*.jsonl" "$OUT/div/test_$V"
  DIV "distilled=$INV_DI" "$OUT/$V/test/distilled_xenc_seed*.jsonl"      "$OUT/div/test_$V"
done

TS="$OUT/div/test_res"; TB="$OUT/div/test_resblind"
RES_HF=$(ros  "$TS" "gemma+ent_humanfact" "_div1.jsonl"); RSB_HF=$(ros  "$TB" "gemma+ent_humanfact" "_div1.jsonl")
RES_JF=$(ros  "$TS" "gemma+ent_xenc"      "_div1.jsonl"); RSB_JF=$(ros  "$TB" "gemma+ent_xenc"      "_div1.jsonl")
RES_HFD=$(ros "$TS" "distilled_humanfact" "_div1.jsonl"); RSB_HFD=$(ros "$TB" "distilled_humanfact" "_div1.jsonl")
RES_JFD=$(ros "$TS" "distilled_xenc"      "_div1.jsonl"); RSB_JFD=$(ros "$TB" "distilled_xenc"      "_div1.jsonl")
RES_HFS=$(ros "$TS" "gemma+ent_humanfactshuffled" "_div1.jsonl")
RES_HFE=$(ros "$TS" "gemma+ent_humanfactempty"    "_div1.jsonl")
RES_JFS=$(ros "$TS" "gemma+ent_xencshuffled"      "_div1.jsonl")
RES_JFE=$(ros "$TS" "gemma+ent_xencempty"         "_div1.jsonl")

echo "===== C2b.2 sealed TEST: the four real arms, raw beside residualised, primary = the RAW MARS-C arm"
EV --acu "$ACU_T" --k 10 --n-boot 2000 --robust-tests --n-perm 10000 --primary "humanfact+div1" \
   --out "$JOB/eval_test_raw_vs_res" --systems \
  "humanfact+div1=$INV_GE:$DIV_HF" "humanfact.res+div1=$INV_GE:$RES_HF" "humanfact.resblind+div1=$INV_GE:$RSB_HF" \
  "judgefact+div1=$INV_GE:$DIV_JF" "judgefact.res+div1=$INV_GE:$RES_JF" "judgefact.resblind+div1=$INV_GE:$RSB_JF" \
  "humanfact.distilled+div1=$INV_DI:$DIV_HFD" "humanfact.distilled.res+div1=$INV_DI:$RES_HFD" "humanfact.distilled.resblind+div1=$INV_DI:$RSB_HFD" \
  "judgefact.distilled+div1=$INV_DI:$DIV_JFD" "judgefact.distilled.res+div1=$INV_DI:$RES_JFD" "judgefact.distilled.resblind+div1=$INV_DI:$RSB_JFD" \
  2>&1 | grep -E "^\[e2 (k|robust)"

echo "===== C2b.3 sealed TEST: the B0 control harness on the residualised score (standing rule of the campaign)"
EV --acu "$ACU_T" --k 10 --n-boot 2000 --primary "humanfact.res+div1" \
   --out "$JOB/eval_test_b0_controls" --systems \
  "humanfact.res+div1=$INV_GE:$RES_HF" "humanfact.shuffled.res+div1=$INV_GE:$RES_HFS" "humanfact.empty.res+div1=$INV_GE:$RES_HFE" \
  "judgefact.res+div1=$INV_GE:$RES_JF" "judgefact.shuffled.res+div1=$INV_GE:$RES_JFS" "judgefact.empty.res+div1=$INV_GE:$RES_JFE" \
  "humanfact+div1=$INV_GE:$DIV_HF" "humanfact.shuffled+div1=$INV_GE:$DIV_HFS" "humanfact.empty+div1=$INV_GE:$DIV_HFE" \
  "judgefact+div1=$INV_GE:$DIV_JF" "judgefact.shuffled+div1=$INV_GE:$DIV_JFS" "judgefact.empty+div1=$INV_GE:$DIV_JFE" \
  2>&1 | grep -E "^\[e2 k"

echo "===== C2b.4 UniSumEval: the same frozen validation surface models applied to the untouched external pool"
for V in res resblind; do
  C2 --mode apply --variant "$V" --arm humanfact --units "gemma+ent=$INV_U" --scores "$RAW_HFU" --model "$M/surface_humanfact.json" --out "$OUT/$V/unisum" --manifest-out "$JOB/apply_unisum_${V}_humanfact.json"
  C2 --mode apply --variant "$V" --arm judgefact --units "gemma+ent=$INV_U" --scores "$RAW_JFU" --model "$M/surface_judgefact.json" --out "$OUT/$V/unisum" --manifest-out "$JOB/apply_unisum_${V}_judgefact.json"
  DIV "gemma+ent=$INV_U" "$OUT/$V/unisum/gemma+ent_humanfact_seed*.jsonl" "$OUT/div/unisum_$V"
  DIV "gemma+ent=$INV_U" "$OUT/$V/unisum/gemma+ent_xenc_seed*.jsonl"      "$OUT/div/unisum_$V"
done
US="$OUT/div/unisum_res"; UB="$OUT/div/unisum_resblind"
RES_HFU=$(ros "$US" "gemma+ent_humanfact" "_div1.jsonl"); RSB_HFU=$(ros "$UB" "gemma+ent_humanfact" "_div1.jsonl")
RES_JFU=$(ros "$US" "gemma+ent_xenc"      "_div1.jsonl"); RSB_JFU=$(ros "$UB" "gemma+ent_xenc"      "_div1.jsonl")
EV --acu "$ACU_U" --k 10 --n-boot 2000 --primary "humanfact+div1" --out "$JOB/eval_unisum_raw_vs_res" --systems \
  "humanfact+div1=$INV_U:$DIV_HFU" "humanfact.res+div1=$INV_U:$RES_HFU" "humanfact.resblind+div1=$INV_U:$RSB_HFU" \
  "judgefact+div1=$INV_U:$DIV_JFU" "judgefact.res+div1=$INV_U:$RES_JFU" "judgefact.resblind+div1=$INV_U:$RSB_JFU" \
  2>&1 | grep -E "^\[e2 k"

(cd "$OUT" && find c2b_heldout -type f -name "*.json" | sort | xargs sha256sum > c2b_heldout/SHA256SUMS)
plan_done "$JOB"
echo "[c2b] done -> $JOB"
