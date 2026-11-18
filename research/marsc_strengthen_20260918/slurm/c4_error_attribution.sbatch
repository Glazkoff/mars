#!/bin/bash
# MARS-C strengthen C4 / H-ATTR -- where does the end-to-end error actually live?
#
# Registered in research/marsc_strengthen_20260918/PREREG.md, Wave C / C4.  No bar: this is the diagnostic
# figure reviewers ask for, and it is decision-relevant before any human is asked to label anything -- if
# INVENTORY misses dominate, the governing proposal's matched-supervision training programme is aimed at the
# wrong stage and the paper should say so.
#
# Every human-marked OMITTED ACU a system fails to surface in its top-10 is attributed to exactly one of
#   inventory  no inventory unit overlaps the ACU under mc_e2_eval.py's own span+content-token rule
#   verifier   an overlapping unit exists but never reaches the budget under the RAW score
#   rule       an overlapping unit is inside the RAW top-10 but the div1 -5.0 demotion pushed it out
# per system, per inventory, per pool, with document-clustered intervals, plus the inventory ceiling.
#
# Self-check printed by the job: `macro recall@10` is recomputed from the same manifests by an independent
# code path and must equal what mc_e2_eval.py reports for that system.  Reconciliation targets for the
# ceiling: gemma+ent 0.8607 validation / 0.8966 test, spaCy 0.4611 (results/mars3/m32/ceilings.json).
#SBATCH --job-name=mars-c4-attribution
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
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[c4] REFUSED -- this job reads the sealed TEST split; needs MC_TEST_REREAD=yes"; exit 3; }
OUT="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}/c2c4"
JOB="$OUT/c4_attribution"; mkdir -p "$JOB"; plan_receipt "$JOB"

G1="$MC/g1_test"; R3="$MC/r03_sumblind"; G7="$MC/g7_unisum"
ACU_V="$G/b18/acu_units_validation.jsonl"; ACU_T="$G/b18/acu_units_test.jsonl"; ACU_U="$G7/acu_units_unisum.jsonl"
IGV="$M3/m32/units_gemma+ent_validation.jsonl"; IDV="$M3/m32/units_distilled_validation.jsonl"
ISV="$M3/m32/units_spacy_validation.jsonl"
IGT="$M3/m32/units_gemma+ent_test.jsonl";       IDT="$M3/m32/units_distilled_test.jsonl"
IGU="$G7/units_gemma+ent_unisum.jsonl"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"

ros () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; \
  [ -f "$f" ] || { echo "[c4] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
C4 () { "$PLAN_PY" "$SCODE/c4_error_attribution.py" --k 10 --n-boot 2000 --expect-seeds 3 --out "$JOB" "$@"; }

# Rosters are built into variables FIRST: an assignment propagates the helper's exit 4 under `set -e`,
# whereas a command substitution used directly as an argument would silently expand to the empty string.
RV_HF=$(ros "$MC/scores" "units_gemma+ent_mc-A-lam1-crossenc-inv" ".jsonl")
DV_HF=$(ros "$MC/e2div/val" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_div1.jsonl")
RV_JF=$(ros "$M3/scores" "units_gemma+ent_xenc" ".jsonl")
DV_JF=$(ros "$MC/e2div_full/val" "units_gemma+ent_xenc" "_div1.jsonl")
RV_HFD=$(ros "$MC/e2imp_all/scores" "distilled_humanfact" ".jsonl")
DV_HFD=$(ros "$MC/e2div/val" "distilled_humanfact" "_div1.jsonl")
RV_JFD=$(ros "$M3/scores" "units_distilled_xenc" ".jsonl")
DV_JFD=$(ros "$MC/e2div/val" "units_distilled_xenc" "_div1.jsonl")
RV_SP=$(ros "$M3/scores" "units_spacy_tagger" ".jsonl")
DV_SP=$(ros "$MC/e2div_full/val" "units_spacy_tagger" "_div1.jsonl")

RT_HF=$(ros "$G1/scores" "gemma+ent_humanfact" ".jsonl"); DT_HF=$(ros "$G1/val" "gemma+ent_humanfact" "_div1.jsonl")
RT_JF=$(ros "$G1/scores" "gemma+ent_xenc" ".jsonl");      DT_JF=$(ros "$G1/val" "gemma+ent_xenc" "_div1.jsonl")
RT_HFD=$(ros "$G1/scores" "distilled_humanfact" ".jsonl"); DT_HFD=$(ros "$G1/val" "distilled_humanfact" "_div1.jsonl")
RT_JFD=$(ros "$G1/scores" "distilled_xenc" ".jsonl");      DT_JFD=$(ros "$G1/val" "distilled_xenc" "_div1.jsonl")
RT_HFS=$(ros "$R3/scores" "gemma+ent_humanfactshuffled" ".jsonl"); DT_HFS=$(ros "$R3/val" "gemma+ent_humanfactshuffled" "_div1.jsonl")
RT_HFE=$(ros "$R3/scores" "gemma+ent_humanfactempty" ".jsonl");    DT_HFE=$(ros "$R3/val" "gemma+ent_humanfactempty" "_div1.jsonl")
RT_JFS=$(ros "$R3/scores" "gemma+ent_xencshuffled" ".jsonl");      DT_JFS=$(ros "$R3/val" "gemma+ent_xencshuffled" "_div1.jsonl")
RT_JFE=$(ros "$R3/scores" "gemma+ent_xencempty" ".jsonl");         DT_JFE=$(ros "$R3/val" "gemma+ent_xencempty" "_div1.jsonl")

RU_HF=$(ros "$G7/scores" "gemma+ent_humanfact" ".jsonl"); DU_HF=$(ros "$G7/val" "gemma+ent_humanfact" "_div1.jsonl")
RU_JF=$(ros "$G7/scores" "gemma+ent_xenc" ".jsonl");      DU_JF=$(ros "$G7/val" "gemma+ent_xenc" "_div1.jsonl")

echo "===== C4.1 validation -- five arms over three inventories"
C4 --acu "$ACU_V" --pool validation --systems \
 "humanfact+div1;$IGV;$RV_HF;$DV_HF" \
 "judgefact+div1;$IGV;$RV_JF;$DV_JF" \
 "humanfact.distilled+div1;$IDV;$RV_HFD;$DV_HFD" \
 "judgefact.distilled+div1;$IDV;$RV_JFD;$DV_JFD" \
 "tagger.spacy+div1;$ISV;$RV_SP;$DV_SP" 2>&1 | grep -E "^\[c4"

echo "===== C4.2 sealed TEST -- the four frozen arms plus the B0 summary-blind controls"
C4 --acu "$ACU_T" --pool test --systems \
 "humanfact+div1;$IGT;$RT_HF;$DT_HF" \
 "judgefact+div1;$IGT;$RT_JF;$DT_JF" \
 "humanfact.distilled+div1;$IDT;$RT_HFD;$DT_HFD" \
 "judgefact.distilled+div1;$IDT;$RT_JFD;$DT_JFD" \
 "humanfact.shuffled+div1;$IGT;$RT_HFS;$DT_HFS" \
 "humanfact.empty+div1;$IGT;$RT_HFE;$DT_HFE" \
 "judgefact.shuffled+div1;$IGT;$RT_JFS;$DT_JFS" \
 "judgefact.empty+div1;$IGT;$RT_JFE;$DT_JFE" 2>&1 | grep -E "^\[c4"

echo "===== C4.3 UniSumEval -- the untouched external pool"
C4 --acu "$ACU_U" --pool unisum --systems \
 "humanfact+div1;$IGU;$RU_HF;$DU_HF" \
 "judgefact+div1;$IGU;$RU_JF;$DU_JF" 2>&1 | grep -E "^\[c4"

(cd "$OUT" && find c4_attribution -type f -name "*.json" | sort | xargs sha256sum > c4_attribution/SHA256SUMS)
plan_done "$JOB"
echo "[c4] done -> $JOB"
