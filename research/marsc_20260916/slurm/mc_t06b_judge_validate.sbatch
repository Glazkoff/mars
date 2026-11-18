#!/bin/bash
# MARS-C T06b -- does the anchored judge clear the 0.85 validity bar against the blinded G6 human labels?
# If it does, judged precision on the emitted units becomes reportable; if it does not, the precision claim stays
# on the human sample alone and this is recorded as such. CPU only.
#SBATCH --job-name=mc-t06b-valid
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
G1="$MC/g1_test"; G7="$MC/g7_unisum"
INV_GE="$M3/m32/units_gemma+ent_test.jsonl"; INV_DI="$M3/m32/units_distilled_test.jsonl"; ACU_T="$G/b18/acu_units_test.jsonl"
INV_GE_V="$M3/m32/units_gemma+ent_validation.jsonl"; INV_DI_V="$M3/m32/units_distilled_validation.jsonl"; ACU_V="$G/b18/acu_units_validation.jsonl"
INV_U="$G7/units_gemma+ent_unisum.jsonl"; ACU_U="$G7/acu_units_unisum.jsonl"

G2A="$MC/g2_anchored"; OUT="$MC/t06_validate"; mkdir -p "$OUT"; plan_receipt "$OUT"
KEY="$MC/g6/human_sample_key.jsonl"; EXPORT="${MC_G6_EXPORT:-$MC/g6/labels_export.jsonl}"
"$PLAN_PY" "$MCCODE/mc_judge_precision.py" analyze --tasks "$MC/g2/tasks.jsonl" --primary "humanfact+div1" --primary-family gemma_anchored --n-boot 300 \
  --families "gemma_anchored=$G2A/judgments/gemma_*.jsonl" "qwen_anchored=$G2A/judgments/qwen_*.jsonl" --out "$OUT" 2>&1 | tail -30
if [ -f "$KEY" ] && [ -f "$EXPORT" ]; then
  "$PLAN_PY" "$MCCODE/mc_judge_vs_human.py" --tasks "$MC/g2/tasks.jsonl" --key "$KEY" --export "$EXPORT" --out "$OUT" \
    --families "gemma_anchored=$G2A/judgments/gemma_*.jsonl" "qwen_anchored=$G2A/judgments/qwen_*.jsonl" \
    "gemma_plain=$MC/g2/judgments/gemma_*.jsonl" "qwen_plain=$MC/g2/judgments/qwen_*.jsonl" 2>&1 | tail -10
else
  echo "[t06b] G6 human export not found (key=$KEY export=$EXPORT) -- judgments produced, human validation pending the annotator export"
fi
plan_done "$OUT"
