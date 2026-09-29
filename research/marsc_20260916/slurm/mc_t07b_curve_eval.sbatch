#!/bin/bash
# MARS-C T07b -- the emission rule and the evaluation over every point of the label-efficiency curve. CPU only.
#SBATCH --job-name=mc-t07b-eval
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
G1="$MC/g1_test"; G7="$MC/g7_unisum"
INV_GE="$M3/m32/units_gemma+ent_test.jsonl"; INV_DI="$M3/m32/units_distilled_test.jsonl"; ACU_T="$G/b18/acu_units_test.jsonl"
INV_GE_V="$M3/m32/units_gemma+ent_validation.jsonl"; INV_DI_V="$M3/m32/units_distilled_validation.jsonl"; ACU_V="$G/b18/acu_units_validation.jsonl"
INV_U="$G7/units_gemma+ent_unisum.jsonl"; ACU_U="$G7/acu_units_unisum.jsonl"

OUT="$MC/t07_curve"; mkdir -p "$OUT/val"; plan_receipt "$OUT"
SPECS=""
for FR in 10 25 50 100; do
  G="$OUT/scores/gemma+ent_t07-frac${FR}_seed*.jsonl"
  [ -n "$(ls $G 2>/dev/null)" ] || { echo "[t07b] no scores for fraction $FR"; continue; }
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "gemma+ent=$INV_GE_V" --scores-glob "$G" --out "$OUT/val"
  SPECS="$SPECS frac${FR}+div1=$INV_GE_V:$OUT/val/gemma+ent_t07-frac${FR}_seed*_div1.jsonl"
done
[ -n "$SPECS" ] || { echo "[t07b] nothing to evaluate"; exit 4; }
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --acu "$ACU_V" --k 5 10 --n-boot 500 --primary "frac100+div1" --out "$OUT/eval_validation" --systems $SPECS 2>&1 | grep "\[e2"
sha256sum "$OUT"/eval_validation/*.json > "$OUT/SHA256SUMS"; plan_done "$OUT"
