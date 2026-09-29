#!/bin/bash
# MARS-C G7a2 -- rebuild the UniSumEval reference rows with a unique pair id (uid:model) and the inventory; archive the collapsed
# 224-pair outputs of the first pass (pair_id = uid collided across the nine summaries of a document). CPU.
#SBATCH --job-name=mc-g7a2
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"; G7="$MC/g7_unisum"
for d in scores val inv eval_unisum; do [ -e "$G7/$d" ] && mv "$G7/$d" "$G7/${d}_collapsed_224"; done
[ -e "$G7/emitted_unisum.jsonl" ] && mv "$G7/emitted_unisum.jsonl" "$G7/emitted_unisum_collapsed_224.jsonl"
mv "$G7/acu_units_unisum.jsonl" "$G7/acu_units_unisum_collapsed_224.jsonl"; mv "$G7/units_gemma+ent_unisum.jsonl" "$G7/units_gemma+ent_unisum_collapsed_224.jsonl"
mkdir -p "$G7/scores" "$G7/val" "$G7/inv"
"$PLAN_PY" "$MCCODE/mc_unisum.py" build --keyfact "$MC/g5/UniSumEval/data/unisumeval_keyfact.jsonl" --our-sources "$G/b18/acu_units_validation.jsonl" "$G/b18/acu_units_test.jsonl" "$MC/e0/docs_train.jsonl" --out "$G7"
"$PLAN_PY" "$MCCODE/mc_unisum.py" inventory --acu "$G7/acu_units_unisum.jsonl" --props "$G7"/props/props_unisum_shard*.jsonl --out "$G7"
