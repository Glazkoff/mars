#!/bin/bash
# B21: zero-shot baselines on the ACU and m2 unit sets (validation). Usage: sbatch --dependency=afterok:<tasks> b21_zeroshot.sbatch
#SBATCH --job-name=g-b21-zs
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/user/results/mars2_gates/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
plan_receipt "$G/b21/receipt_zeroshot"
SPLIT="${B21_SPLIT:-validation}"
"$PLAN_PY" "$CODE/b21_zeroshot.py" --unit-set acu --units "$G/b18/acu_units_${SPLIT}.jsonl" --systems "${B21_SYSTEMS:-lex,minilm,minicheck,alignscore,nli,summac_zs}" --out "$G/b21/scores" --batch 32
"$PLAN_PY" "$CODE/b21_zeroshot.py" --unit-set m2 --units "$B1_DIR/labels_${SPLIT}.jsonl" --systems "${B21_SYSTEMS:-lex,minilm,minicheck,alignscore,nli,summac_zs}" --out "$G/b21/scores" --batch 32
plan_done "$G/b21/receipt_zeroshot"
