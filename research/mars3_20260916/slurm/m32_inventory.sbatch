#!/bin/bash
# M32: LLM inventories + ceilings after the decomposition (CPU). Env: M32_VARIANTS (default "gemma gemma+ent"), M32_PROPS glob.
#SBATCH --job-name=m3-inventory
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
PROPS=$(ls ${M32_PROPS:-$M3/m32/props_shard*.jsonl})
for V in ${M32_VARIANTS:-gemma gemma+ent}; do for S in validation test; do "$PLAN_PY" "$M3CODE/m32_inventory.py" --split "$S" --variant "$V" --props $PROPS; done; done
cat "$M3/m32/ceilings.json" | head -60
