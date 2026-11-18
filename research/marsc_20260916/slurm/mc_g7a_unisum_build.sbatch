#!/bin/bash
# MARS-C G7a -- UniSumEval external confirmation: build the key-fact reference rows and the sentence decomposition tasks. CPU.
#SBATCH --job-name=mc-g7a
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"; G7="$MC/g7_unisum"; mkdir -p "$G7/props" "$G7/scores" "$G7/val" "$G7/inv"; plan_receipt "$G7"
"$PLAN_PY" "$MCCODE/mc_unisum.py" build --keyfact "$MC/g5/UniSumEval/data/unisumeval_keyfact.jsonl" --our-sources "$G/b18/acu_units_validation.jsonl" "$G/b18/acu_units_test.jsonl" "$MC/e0/docs_train.jsonl" --out "$G7"
