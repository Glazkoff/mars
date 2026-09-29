#!/bin/bash
# MARS-C E2 pilot (validation, exploratory): human-fact-trained verifier (arm A, cross-encoder) scores the frozen gemma+ent
# inventory of every validation pair and saves the model; 3 seeds.
#SBATCH --job-name=mc-e2pilot
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/user/results/marsc/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; export MCCODE="$PLAN_REPO/research/marsc_20260916/code"
read -r -a SEEDS <<< "${MC_SEEDS:-20260906 20260907 20260908}"; SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
INV="$M3/m32/units_gemma+ent_validation.jsonl"; [ -s "$INV" ] || { echo "no inventory"; exit 3; }
TAG="mc-A-lam1-crossenc-inv"; OUT="$MC/e1/${TAG}_seed${SEED}"; mkdir -p "$OUT"; plan_receipt "$OUT"
"$PLAN_PY" "$MCCODE/mc_train.py" --arm A --lam 1 --repr crossenc --seed "$SEED" --data-dir "$MC/e0" --out "$OUT" --tag "$TAG" --scores-out "$MC/scores" --eval-units "units_gemma+ent=$INV" --save-model
plan_done "$OUT"
