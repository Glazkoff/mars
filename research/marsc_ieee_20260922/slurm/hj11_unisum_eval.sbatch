#!/bin/bash
# H-J11 (gap fill, 2026-09-27) -- UniSumEval recall@10 for the Table 1 rows that had none: MARS-2 (released
# ranking and rule), judge-label distilled (seed-avg and consensus), human-fact distilled consensus. One evaluator
# call with the R01.4 conventions (--strict-seeds, explicit three-seed manifests, primary humanfact+div1 on the frozen
# Gemma+ent inventory) so the new rows share the bootstrap of the published ones; the primary must reproduce 0.347.
#SBATCH --job-name=mars-hj11-unisum-ev
#SBATCH --partition=infer
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%j.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
G7="$MC/g7_unisum"; D="$MCS/hj11/unisum_distilled"; M2="$MCS/hj11/unisum_mars2"; OUT="$MCS/hj11/unisum_eval"
mkdir -p "$OUT/inv" "$OUT/val"; plan_receipt "$OUT"
INV_U="$G7/units_gemma+ent_unisum.jsonl"; ACU_U="$G7/acu_units_unisum.jsonl"; INV_DU="$D/units_distilled_unisum.jsonl"
SEEDS="20260906 20260907 20260908"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || { echo "[hj11] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
"$PLAN_PY" "$MCCODE/mc_variants.py" dumps2units --dumps "$M2/direct_enabled_seed*/units_unisum.jsonl" \
  --source-units "$ACU_U" --name mars2_direct_enabled --split unisum --out "$OUT/inv"
"$PLAN_PY" "$MCCODE/mc_diversity.py" --units "mars2_direct_enabled=$OUT/inv/units_mars2_direct_enabled_unisum.jsonl" \
  --scores-glob "$OUT/inv/mars2_direct_enabled_mars2_seed0.jsonl" --out "$OUT/val"
M2U="$OUT/inv/units_mars2_direct_enabled_unisum.jsonl"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU_U" --k 10 --n-boot 500 \
  --primary "humanfact+div1" --out "$OUT/eval" --systems \
  "humanfact+div1=$INV_U:$(roster "$G7/val" gemma+ent_humanfact _div1.jsonl)" \
  "humanfact.distilled+div1=$INV_DU:$(roster "$D/val" distilled_humanfact _div1.jsonl)" \
  "judgefact.distilled+div1=$INV_DU:$(roster "$D/val" distilled_xenc _div1.jsonl)" \
  "judgefact.distilled.single+div1=$INV_DU:$D/val/distilled_xenc_seedavg_div1.jsonl" \
  "mars2.direct_enabled=$M2U:$OUT/inv/mars2_direct_enabled_mars2_seed0.jsonl" \
  "mars2.direct_enabled+div1=$M2U:$OUT/val/mars2_direct_enabled_mars2_seed0_div1.jsonl" 2>&1 | grep -E "\[e2"
(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[hj11] UniSumEval eval done -> $OUT/eval"
