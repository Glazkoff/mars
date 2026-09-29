#!/bin/bash
# MARS-C E1 arm runs. Env: MC_ARMS ("R A B C D"), MC_LAMS ("1"), MC_SEEDS, MC_RESOURCES (comma; empty = all), MC_REPR (auto|tagger|crossenc), MC_TAGSFX.
# Task index -> (arm, lam, seed) in that nesting. Exits 3 if the E0 gate failed.
#SBATCH --job-name=mc-e1
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; export MCCODE="$PLAN_REPO/research/marsc_20260916/code"; mkdir -p "$MC/logs" "$MC/e0" "$MC/e1" "$MC/scores" "$MC/analysis"
read -r -a ARMS <<< "${MC_ARMS:-R A B C D}"; read -r -a LAMS <<< "${MC_LAMS:-1}"; read -r -a SEEDS <<< "${MC_SEEDS:-20260906 20260907 20260908}"
NA=${#ARMS[@]}; NL=${#LAMS[@]}; NS=${#SEEDS[@]}; T="${SLURM_ARRAY_TASK_ID:-0}"
ARM="${ARMS[$((T / (NL * NS)))]}"; LAM="${LAMS[$(((T / NS) % NL))]}"; SEED="${SEEDS[$((T % NS))]}"
GATE=$("$PLAN_PY" -c "import json; g=json.load(open('$MC/e0/census.json')); print('PASS' if g['gate']['PASS'] else 'FAIL', g.get('collisions_train',{}).get('rate',0))")
set -- $GATE; [ "$1" = PASS ] || { echo "E0 gate failed: $GATE"; exit 3; }
REPR="${MC_REPR:-auto}"; if [ "$REPR" = auto ]; then REPR=$("$PLAN_PY" -c "print('tagger' if float('$2') <= 0.05 else 'crossenc')"); fi
TAG="mc-${ARM}-lam${LAM}-${REPR}${MC_TAGSFX:-}"; OUT="$MC/e1/${TAG}_seed${SEED}"; mkdir -p "$OUT"; plan_receipt "$OUT"
"$PLAN_PY" "$MCCODE/mc_train.py" --arm "$ARM" --lam "$LAM" --repr "$REPR" --seed "$SEED" --data-dir "$MC/e0" --out "$OUT" --tag "$TAG" --scores-out "$MC/scores" ${MC_RESOURCES:+--resources "$MC_RESOURCES"} ${MC_STEPS:+--steps "$MC_STEPS"}
plan_done "$OUT"
