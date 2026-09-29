#!/bin/bash
# MARS-C R12b -- omission precision for the remaining table cells (MARS-2 arms and the two single-model cells
# R12 did not carry), so every row of the main table reports both endpoints.
#SBATCH --job-name=mc-r12b-fill
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[r12b] REFUSED"; exit 3; }
G1="$MC/g1_test"; T5="$MC/t05_mars2"; OUT="$MC/r12_precision"; mkdir -p "$OUT"
INV_GE="$M3/m32/units_gemma+ent_test.jsonl"; INV_DI="$M3/m32/units_distilled_test.jsonl"; ACU_T="$G/b18/acu_units_test.jsonl"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || exit 4; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU_T" --k 10 --n-boot 2000 \
  --primary "humanfact+div1" --out "$OUT/eval_test_fill" --systems \
  "humanfact+div1=$INV_GE:$(roster "$G1/val" "gemma+ent_humanfact" "_div1.jsonl")" \
  "humanfact.distilled.single+div1=$INV_DI:$G1/val/distilled_humanfact_seedavg_div1.jsonl" \
  "judgefact.single+div1=$INV_GE:$G1/val/gemma+ent_xenc_seedavg_div1.jsonl" \
  "humanfact=$INV_GE:$(roster "$G1/scores" "gemma+ent_humanfact" ".jsonl")" \
  "judgefact=$INV_GE:$(roster "$G1/scores" "gemma+ent_xenc" ".jsonl")" \
  "humanfact.distilled=$INV_DI:$(roster "$G1/scores" "distilled_humanfact" ".jsonl")" \
  "judgefact.distilled=$INV_DI:$(roster "$G1/scores" "distilled_xenc" ".jsonl")" \
  "mars2.full_recovery=$T5/inv/units_mars2_full_recovery_test.jsonl:$T5/inv/mars2_full_recovery_mars2_seed0.jsonl" \
  "mars2.full_recovery+div1=$T5/inv/units_mars2_full_recovery_test.jsonl:$T5/val/mars2_full_recovery_mars2_seed0_div1.jsonl" \
  "mars2.direct_enabled=$T5/inv/units_mars2_direct_enabled_test.jsonl:$T5/inv/mars2_direct_enabled_mars2_seed0.jsonl" \
  "mars2.direct_enabled+div1=$T5/inv/units_mars2_direct_enabled_test.jsonl:$T5/val/mars2_direct_enabled_mars2_seed0_div1.jsonl" 2>&1 | grep -E "^\[e2 k"
echo "[r12b] done"
