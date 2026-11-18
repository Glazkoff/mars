#!/bin/bash
# MARS-C T07 -- end-to-end label-efficiency curve of the verifier: the E1b arm A cross-encoder retrained on
# 10 / 25 / 50 / 100 % of the human-fact TRAIN documents x 3 seeds, then scored on the validation inventory.
# Answers 'how much human annotation does this need' and bounds the cost of adopting the method. VALIDATION only.
#SBATCH --job-name=mc-t07-curve
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/home/user/results/marsc/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
G1="$MC/g1_test"; G7="$MC/g7_unisum"
INV_GE="$M3/m32/units_gemma+ent_test.jsonl"; INV_DI="$M3/m32/units_distilled_test.jsonl"; ACU_T="$G/b18/acu_units_test.jsonl"
INV_GE_V="$M3/m32/units_gemma+ent_validation.jsonl"; INV_DI_V="$M3/m32/units_distilled_validation.jsonl"; ACU_V="$G/b18/acu_units_validation.jsonl"
INV_U="$G7/units_gemma+ent_unisum.jsonl"; ACU_U="$G7/acu_units_unisum.jsonl"

OUT="$MC/t07_curve"; mkdir -p "$OUT/scores"
FRACS=(10 25 50 100); read -r -a SEEDS <<< "${MC_SEEDS:-20260906 20260907 20260908}"; NS=${#SEEDS[@]}
T="${SLURM_ARRAY_TASK_ID:-0}"; FR="${FRACS[$((T / NS))]}"; SEED="${SEEDS[$((T % NS))]}"
NTRAIN=$("$PLAN_PY" -c "import json;g=json.load(open('$MC/e0/census.json'));n=int(g['roles']['train']['docs']);print(max(8,round(n*$FR/100)))")
TAG="t07-frac${FR}"; RUN="$OUT/${TAG}_seed${SEED}"; mkdir -p "$RUN"; plan_receipt "$RUN"
echo "[t07] fraction ${FR}% -> $NTRAIN train documents, seed $SEED"
"$PLAN_PY" "$MCCODE/mc_train.py" --arm A --lam 1 --repr crossenc --seed "$SEED" --data-dir "$MC/e0" --out "$RUN" --tag "$TAG" \
  --train-limit "$NTRAIN" --scores-out "$OUT/scores" --eval-units "gemma+ent=$INV_GE_V" --eval-acu "$ACU_V"
plan_done "$RUN"
