#!/bin/bash
# MARS-C R07a -- complete the crossed inventory x verifier design on validation, then build its judge tasks.
#
# The G3 validation run never evaluated (gemma+ent inventory, judge-label verifier), so the emitted dump the
# blinded human sample was drawn from is missing exactly the cell that identifies the inventory main effect the
# paper asserts. The per-seed scores for it exist (e2div_full); this job evaluates all four cells at k=10 from
# explicit three-seed manifests, dumps their top-10 emissions, and builds the coverage-judge tasks over them.
#SBATCH --job-name=mc-r07a-build
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
OUT="$MC/r07_crossed"; mkdir -p "$OUT/judgments"; plan_receipt "$OUT"
DIV="$MC/e2div"; DF="$MC/e2div_full"; G3="$MC/g3"
INV_GE="$M3/m32/units_gemma+ent_validation.jsonl"; INV_DI="$M3/m32/units_distilled_validation.jsonl"
INV_M2="$G3/inv/units_mars2_validation.jsonl"; ACU_V="$G/b18/acu_units_validation.jsonl"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || { echo "[r07a] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }

echo "===== R07a.1 all four crossed cells on validation, top-10 emissions dumped"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU_V" --k 10 --n-boot 500 \
  --primary "humanfact+div1" --dump-emitted "$OUT/emitted_validation_k10.jsonl" --out "$OUT/eval_validation" --systems \
  "humanfact+div1=$INV_GE:$(roster "$DIV/val" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_div1.jsonl")" \
  "judgefact+div1=$INV_GE:$(roster "$DF/val" "units_gemma+ent_xenc" "_div1.jsonl")" \
  "humanfact.distilled+div1=$INV_DI:$(roster "$DIV/val" "distilled_humanfact" "_div1.jsonl")" \
  "judgefact.distilled+div1=$INV_DI:$(roster "$DIV/val" "units_distilled_xenc" "_div1.jsonl")" \
  "mars2=$INV_M2:$G3/inv/mars2_mars2_seed0.jsonl" 2>&1 | grep -E "\[e2"

echo "===== R07a.2 coverage-judge tasks over the top-10 emissions of all five systems"
"$PLAN_PY" "$MCCODE/mc_judge_precision.py" build --emitted "$OUT/emitted_validation_k10.jsonl" \
  --acu "$ACU_V" --max-rank 10 \
  --systems "humanfact+div1" "judgefact+div1" "humanfact.distilled+div1" "judgefact.distilled+div1" "mars2" \
  --out "$OUT"
wc -l "$OUT/tasks.jsonl"
