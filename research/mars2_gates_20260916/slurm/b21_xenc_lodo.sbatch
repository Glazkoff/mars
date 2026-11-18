#!/bin/bash
# B21b: cross-resource folds of the supervised cross-encoder (each B1 resource held out; 3 seeds). Registered 2026-09-16 (PREREG amendment).
#SBATCH --job-name=g-b21-xenc-lodo
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=/home/user/results/mars2_gates/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
DOMS=(rose_cnndm rose_xsum rose_samsum natural770); read -r -a SEEDS <<< "${B21_SEEDS:-20260906 20260907 20260908}"; NS=${#SEEDS[@]}
T="${SLURM_ARRAY_TASK_ID:-0}"; D="${DOMS[$((T / NS))]}"; SEED="${SEEDS[$((T % NS))]}"
OUT="$G/b21/xenc_xdom-${D}_seed${SEED}"; mkdir -p "$OUT"; plan_receipt "$OUT"
"$PLAN_PY" "$CODE/b21_finetune_xenc.py" --seed "$SEED" --exclude-resource "$D" --tag "xdom-${D}" --out "$OUT" --eval-acu "$G/b18/acu_units_validation.jsonl" --scores-out "$G/b21/scores"
rm -rf "$OUT/model.safetensors" "$OUT"/*.bin 2>/dev/null || true    # keep the receipt + config, not 1.6 GB per fold
plan_done "$OUT"
