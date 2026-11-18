#!/bin/bash
# M32: sentence tasks for the LLM decomposition + the spaCy inventory of every validation/test pair (CPU).
#SBATCH --job-name=m3-m32-tasks
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user/results/mars3/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MARS2_DEVICE=cpu
"$PLAN_PY" "$M3CODE/m32_decompose.py" --build-tasks
for S in validation test; do "$PLAN_PY" "$M3CODE/m32_inventory.py" --split "$S" --variant spacy; done
