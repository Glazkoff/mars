#!/bin/bash
# MARS-C T06 -- re-validation of the precision judge with boundary-anchored prompts (four hand-written worked
# examples, no corpus text). The plain prompts scored 0.685 / 0.705 against the human labels, below the registered
# 0.85 validity bar, so judged precision on the 69,596 emitted units is currently unusable. Two families x 3 shards.
#SBATCH --job-name=mc-t06-judge
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/user/results/marsc/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
G1="$MC/g1_test"; G7="$MC/g7_unisum"
INV_GE="$M3/m32/units_gemma+ent_test.jsonl"; INV_DI="$M3/m32/units_distilled_test.jsonl"; ACU_T="$G/b18/acu_units_test.jsonl"
INV_GE_V="$M3/m32/units_gemma+ent_validation.jsonl"; INV_DI_V="$M3/m32/units_distilled_validation.jsonl"; ACU_V="$G/b18/acu_units_validation.jsonl"
INV_U="$G7/units_gemma+ent_unisum.jsonl"; ACU_U="$G7/acu_units_unisum.jsonl"

G2A="$MC/g2_anchored"; mkdir -p "$G2A/judgments"
[ -f "$MC/g2/tasks.jsonl" ] || { echo "[t06] missing $MC/g2/tasks.jsonl (the G2 build stage)"; exit 3; }
T="${SLURM_ARRAY_TASK_ID:-0}"; FAMS=(gemma qwen); MODELS=("google/gemma-4-31B-it" "Qwen/Qwen3.8-27B")
FAM="${FAMS[$((T / 3))]}"; SHARD="$((T % 3))/3"
SNAP=$("$PLAN_PY" -c "import sys; sys.path.insert(0,'$PLAN_REPO/research/mars3_20260916/code'); import m3common as m; print(m.snap('${MODELS[$((T / 3))]}'))")
echo "[t06] task $T family $FAM shard $SHARD snapshot $SNAP anchored"; plan_receipt "$G2A/receipt_${FAM}_$((T % 3))"
for Q in coverage support; do
  "$PLAN_PY" "$MCCODE/mc_judge_precision.py" run --tasks "$MC/g2/tasks.jsonl" --out "$G2A/judgments/${FAM}_${Q}_shard$((T % 3)).jsonl" \
    --model-snapshot "$SNAP" --question "$Q" --shard "$SHARD" --batch-size 8 --prompt-set anchored
done
plan_done "$G2A/receipt_${FAM}_$((T % 3))"
