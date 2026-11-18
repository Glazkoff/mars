#!/bin/bash
# MARS-C R09b -- the importance-policy cells the paper's "lifts an unruled emitter, costs once the rule is
# applied" sentence rests on, recomputed from explicit three-seed manifests (they were compared against a
# contaminated reference cell). Policy alone, rule alone, policy-then-rule, and the two registered controls.
#SBATCH --job-name=mc-r09b-policy
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
DIV="$MC/e2div"; IMP2="$MC/e2imp_all"; CTL="$MC/e2ctl"; OUT="$MC/r09_validation"; mkdir -p "$OUT"
INV_GE="$M3/m32/units_gemma+ent_validation.jsonl"; INV_DI="$M3/m32/units_distilled_validation.jsonl"
ACU_V="$G/b18/acu_units_validation.jsonl"; SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || { echo "[r09b] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU_V" --k 5 10 --n-boot 500 \
  --primary "humanfact+div1" --out "$OUT/eval_policy" --systems \
  "humanfact+div1=$INV_GE:$(roster "$DIV/val" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_div1.jsonl")" \
  "humanfact+imp+div1=$INV_GE:$(roster "$DIV/val" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_imp0.2_div1.jsonl")" \
  "humanfact+imp=$INV_GE:$(roster "$IMP2/val" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_imp0.2.jsonl")" \
  "humanfact=$INV_GE:$(roster "$MC/scores" "units_gemma+ent_mc-A-lam1-crossenc-inv" ".jsonl")" \
  "humanfact+leadpos=$INV_GE:$(roster "$CTL/val_lead" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_imp0.2.jsonl")" \
  "humanfact+random=$INV_GE:$(roster "$CTL/val_rand" "units_gemma+ent_mc-A-lam1-crossenc-inv" "_imp0.795.jsonl")" \
  "humanfact.distilled+div1=$INV_DI:$(roster "$DIV/val" "distilled_humanfact" "_div1.jsonl")" \
  "humanfact.distilled+imp+div1=$INV_DI:$(roster "$DIV/val" "distilled_humanfact" "_imp0.2_div1.jsonl")" \
  "humanfact.distilled+imp=$INV_DI:$(roster "$IMP2/val" "distilled_humanfact" "_imp0.2.jsonl")" \
  "judgefact.distilled+div1=$INV_DI:$(roster "$DIV/val" "units_distilled_xenc" "_div1.jsonl")" \
  "judgefact.distilled+imp+div1=$INV_DI:$(roster "$DIV/val" "units_distilled_xenc" "_imp0.2_div1.jsonl")" 2>&1 | grep -E "\[e2"
echo "[r09b] done"
