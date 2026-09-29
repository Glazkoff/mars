#!/bin/bash
# H-J11 (gap fill, 2026-09-27) -- the distilled-inventory rows of Table 1 on UniSumEval: the frozen LongT5 segmenter
# over the frozen UniSumEval sentence tasks, a propositions-only inventory (new file; the Gemma+ent one is untouched),
# then the frozen verifiers exactly as G1 scored the RoSE distilled inventory: human-fact (3 seeds), judge-label xenc
# (3 seeds), the div1 rule per seed file, and the judge-label seed average for the seed-avg row.
#SBATCH --job-name=mars-hj11-unisum-di
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%j.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
JCODE="$PLAN_REPO/research/marsc_ieee_20260922/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
G7="$MC/g7_unisum"; OUT="$MCS/hj11/unisum_distilled"; mkdir -p "$OUT/scores" "$OUT/inv" "$OUT/val"; plan_receipt "$OUT"
INV="$OUT/units_distilled_unisum.jsonl"
[ -s "$OUT/props_distilled_unisum.jsonl" ] || "$PLAN_PY" "$JCODE/hj11_unisum_distilled.py" segment \
  --tasks "$G7/decompose_tasks_unisum.jsonl" --segmenter "$M3/m32/segmenter" --out "$OUT/props_distilled_unisum.jsonl"
"$PLAN_PY" "$JCODE/hj11_unisum_distilled.py" inventory --acu "$G7/acu_units_unisum.jsonl" \
  --props "$OUT/props_distilled_unisum.jsonl" --out "$INV"
"$PLAN_PY" "$MCCODE/mc_score_units.py" --units "distilled=$INV" --models "$MC/e1/mc-A-lam1-crossenc-inv_seed*/model" \
  --tag humanfact --scores-out "$OUT/scores"
"$PLAN_PY" "$M3CODE/m31_score_pipeline_all.py" --units "distilled=$INV" --ckpts "$G/b21/xenc_seed*" --scores-out "$OUT/scores"
ls -la "$OUT/scores"
for f in "$OUT"/scores/distilled_humanfact_seed*.jsonl "$OUT"/scores/distilled_xenc_seed*.jsonl; do
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "distilled=$INV" --scores-glob "$f" --out "$OUT/val"
done
"$PLAN_PY" "$MCCODE/mc_variants.py" seedavg --units "distilled=$INV" --scores "$OUT/scores/distilled_xenc_seed*.jsonl" --tag xenc --out "$OUT/inv"
"$PLAN_PY" "$MCCODE/mc_diversity.py" --units "distilled=$INV" --scores-glob "$OUT/inv/distilled_xenc_seedavg.jsonl" --out "$OUT/val"
ls -la "$OUT/val"
(cd "$OUT" && find . -maxdepth 2 -type f -name "*.jsonl" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[hj11] distilled UniSumEval done -> $OUT"
