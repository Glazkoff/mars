#!/bin/bash
# MARS-C R12 -- the emitted-omission PRECISION endpoint, added because R03 showed recall@k alone is
# summary-blind-gameable.
#
# R03b/c rescored the frozen inventories with another document's summary, or none, and recall@10 went UP
# (0.377 -> 0.440 shuffled, 0.458 fact-only). The mechanism is not that the system ignores the summary --- R03a
# shows the verifier picks the omitting summary over the conveying one for the same fact 92% of the time ---
# but that recall@k never charges for demoting a fact the summary DID convey. A summary-blind emitter demotes
# nothing, aligns to more reference ACUs (0.226 vs 0.173 of its emissions), and is rewarded for it.
#
# The endpoint that charges for it is: among emitted units that align to SOME reference ACU, the share aligning
# to one the annotators marked OMITTED. This job recomputes every headline cell with that quantity beside
# recall, on both pools, from the same explicit three-seed manifests.
#SBATCH --job-name=mc-r12-prec
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[r12] REFUSED"; exit 3; }
G1="$MC/g1_test"; G7="$MC/g7_unisum"; R3="$MC/r03_sumblind"; OUT="$MC/r12_precision"; mkdir -p "$OUT"; plan_receipt "$OUT"
INV_GE="$M3/m32/units_gemma+ent_test.jsonl"; INV_DI="$M3/m32/units_distilled_test.jsonl"; ACU_T="$G/b18/acu_units_test.jsonl"
INV_U="$G7/units_gemma+ent_unisum.jsonl"; ACU_U="$G7/acu_units_unisum.jsonl"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || { echo "[r12] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
EV () { "$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 "$@"; }

echo "===== R12.1 held-out TEST: every frozen cell plus the summary-blind arms, recall AND omission precision"
EV --acu "$ACU_T" --k 10 --n-boot 2000 --primary "humanfact+div1" --out "$OUT/eval_test" --systems \
  "humanfact+div1=$INV_GE:$(roster "$G1/val" "gemma+ent_humanfact" "_div1.jsonl")" \
  "judgefact+div1=$INV_GE:$(roster "$G1/val" "gemma+ent_xenc" "_div1.jsonl")" \
  "humanfact.distilled+div1=$INV_DI:$(roster "$G1/val" "distilled_humanfact" "_div1.jsonl")" \
  "judgefact.distilled+div1=$INV_DI:$(roster "$G1/val" "distilled_xenc" "_div1.jsonl")" \
  "humanfact.single+div1=$INV_GE:$G1/val/gemma+ent_humanfact_seedavg_div1.jsonl" \
  "judgefact.distilled.single+div1=$INV_DI:$G1/val/distilled_xenc_seedavg_div1.jsonl" \
  "humanfact.shuffled+div1=$INV_GE:$(roster "$R3/val" "gemma+ent_humanfactshuffled" "_div1.jsonl")" \
  "humanfact.empty+div1=$INV_GE:$(roster "$R3/val" "gemma+ent_humanfactempty" "_div1.jsonl")" \
  "judgefact.shuffled+div1=$INV_GE:$(roster "$R3/val" "gemma+ent_xencshuffled" "_div1.jsonl")" \
  "judgefact.empty+div1=$INV_GE:$(roster "$R3/val" "gemma+ent_xencempty" "_div1.jsonl")" \
  "humanfact.distilled.shuffled+div1=$INV_DI:$(roster "$R3/val" "distilled_humanfactshuffled" "_div1.jsonl")" \
  "humanfact.distilled.empty+div1=$INV_DI:$(roster "$R3/val" "distilled_humanfactempty" "_div1.jsonl")" \
  "judgefact.distilled.shuffled+div1=$INV_DI:$(roster "$R3/val" "distilled_xencshuffled" "_div1.jsonl")" \
  "judgefact.distilled.empty+div1=$INV_DI:$(roster "$R3/val" "distilled_xencempty" "_div1.jsonl")" 2>&1 | grep -E "^\[e2 k"

echo "===== R12.2 UniSumEval: the same endpoint on the untouched pool"
EV --acu "$ACU_U" --k 10 --n-boot 2000 --primary "humanfact+div1" --out "$OUT/eval_unisum" --systems \
  "humanfact+div1=$INV_U:$(roster "$G7/val" "gemma+ent_humanfact" "_div1.jsonl")" \
  "judgefact+div1=$INV_U:$(roster "$G7/val" "gemma+ent_xenc" "_div1.jsonl")" \
  "humanfact.single+div1=$INV_U:$G7/val/gemma+ent_humanfact_seedavg_div1.jsonl" \
  "judgefact.single+div1=$INV_U:$G7/val/gemma+ent_xenc_seedavg_div1.jsonl" 2>&1 | grep -E "^\[e2 k"
(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[r12] done"
