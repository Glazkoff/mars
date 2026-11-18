#!/bin/bash
# H-J13 (2026-09-29, registered in research/mars_metric_20260923/PREREG.md, Amendment M-H (c)): the random-filter and
# lead-position controls of the emission rule, like for like. The R09 repair already evaluated both filters from the
# explicit three-seed manifests but without the rule; this job applies the frozen rule (cap 1) on top of each filter
# from the identical files and evaluates filter, filter+rule, rule and raw ranking in one call. CPU only.
#SBATCH --job-name=mars-hj13-rulectl
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
DIV="$MC/e2div"; CTL="$MC/e2ctl"; OUT="/home/user/results/marsc_strengthen/hj13"; mkdir -p "$OUT/val"; plan_receipt "$OUT"
INV_GE="$M3/m32/units_gemma+ent_validation.jsonl"; ACU_V="$G/b18/acu_units_validation.jsonl"
SEEDS="20260906 20260907 20260908"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || { echo "[hj13] MISSING $f" >&2; return 1; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
PRE="units_gemma+ent_mc-A-lam1-crossenc-inv"
# the rule on top of each filter, from the very files R09 evaluated
"$PLAN_PY" "$MCCODE/mc_diversity.py" --units "gemma+ent=$INV_GE" --scores-glob "$CTL/val_lead/${PRE}_seed*_imp0.2.jsonl" --out "$OUT/val"
"$PLAN_PY" "$MCCODE/mc_diversity.py" --units "gemma+ent=$INV_GE" --scores-glob "$CTL/val_rand/${PRE}_seed*_imp0.795.jsonl" --out "$OUT/val"
HF_RULE=$(roster "$DIV/val" "$PRE" "_div1.jsonl")
HF_RAW=$(roster "$MC/scores" "$PRE" ".jsonl")
LEAD=$(roster "$CTL/val_lead" "$PRE" "_imp0.2.jsonl")
RAND=$(roster "$CTL/val_rand" "$PRE" "_imp0.795.jsonl")
LEAD_RULE=$(roster "$OUT/val" "$PRE" "_imp0.2_div1.jsonl")
RAND_RULE=$(roster "$OUT/val" "$PRE" "_imp0.795_div1.jsonl")
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU_V" --k 10 --n-boot 500 \
  --primary "humanfact+div1" --out "$OUT/eval_validation" --systems \
  "humanfact+div1=$INV_GE:$HF_RULE" \
  "humanfact=$INV_GE:$HF_RAW" \
  "humanfact+leadpos=$INV_GE:$LEAD" \
  "humanfact+leadpos+div1=$INV_GE:$LEAD_RULE" \
  "humanfact+random=$INV_GE:$RAND" \
  "humanfact+random+div1=$INV_GE:$RAND_RULE" 2>&1 | grep -E "\[e2"
(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[hj13] done -> $OUT"
