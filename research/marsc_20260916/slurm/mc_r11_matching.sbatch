#!/bin/bash
# MARS-C R11 -- is the result an artefact of the lenient match rule? (FULL_REVIEW item 6.)
#
# An emitted unit is credited with recovering an ACU when its source span overlaps the ACU's best-matching
# source sentence AND the two share at least one content token. That lexical status agrees with the human Q2
# answer on only 0.71--0.72 of sampled units, so the rule is the weakest link in the protocol. This job re-runs
# the held-out test evaluation under progressively stricter rules -- two and three shared content tokens, and a
# content-token Jaccard floor -- applied IDENTICALLY to every system. Absolute recall must fall; what matters is
# whether the human-fact minus judge-label margin survives.
#SBATCH --job-name=mc-r11-matching
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
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[r11] REFUSED"; exit 3; }
G1="$MC/g1_test"; OUT="$MC/r11_matching"; mkdir -p "$OUT"; plan_receipt "$OUT"
INV_GE="$M3/m32/units_gemma+ent_test.jsonl"; INV_DI="$M3/m32/units_distilled_test.jsonl"; ACU_T="$G/b18/acu_units_test.jsonl"
INV_U="$MC/g7_unisum/units_gemma+ent_unisum.jsonl"; ACU_U="$MC/g7_unisum/acu_units_unisum.jsonl"; G7="$MC/g7_unisum"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || { echo "[r11] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
SPECS_T="humanfact+div1=$INV_GE:$(roster "$G1/val" "gemma+ent_humanfact" "_div1.jsonl") \
judgefact.distilled+div1=$INV_DI:$(roster "$G1/val" "distilled_xenc" "_div1.jsonl") \
humanfact.distilled+div1=$INV_DI:$(roster "$G1/val" "distilled_humanfact" "_div1.jsonl") \
judgefact+div1=$INV_GE:$(roster "$G1/val" "gemma+ent_xenc" "_div1.jsonl") \
humanfact.single+div1=$INV_GE:$G1/val/gemma+ent_humanfact_seedavg_div1.jsonl \
judgefact.distilled.single+div1=$INV_DI:$G1/val/distilled_xenc_seedavg_div1.jsonl"
SPECS_U="humanfact+div1=$INV_U:$(roster "$G7/val" "gemma+ent_humanfact" "_div1.jsonl") \
judgefact+div1=$INV_U:$(roster "$G7/val" "gemma+ent_xenc" "_div1.jsonl") \
humanfact.single+div1=$INV_U:$G7/val/gemma+ent_humanfact_seedavg_div1.jsonl \
judgefact.single+div1=$INV_U:$G7/val/gemma+ent_xenc_seedavg_div1.jsonl"
for CFG in "1 0.0 t1" "2 0.0 t2" "3 0.0 t3" "1 0.20 j20" "1 0.34 j34"; do
  set -- $CFG
  echo "===== R11 match-min-tokens=$1 match-min-jaccard=$2"
  "$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --match-min-tokens "$1" --match-min-jaccard "$2" \
    --acu "$ACU_T" --k 10 --n-boot 500 --primary "humanfact+div1" --out "$OUT/eval_test_$3" --systems $SPECS_T 2>&1 | grep -E "^\[e2 k"
  "$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --match-min-tokens "$1" --match-min-jaccard "$2" \
    --acu "$ACU_U" --k 10 --n-boot 500 --primary "humanfact+div1" --out "$OUT/eval_unisum_$3" --systems $SPECS_U 2>&1 | grep -E "^\[e2 k"
done
(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[r11] done"
