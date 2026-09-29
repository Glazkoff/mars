#!/bin/bash
# M31: label-free extractive pseudo rows from ALL train sources (pre-training data for tagger_pre); CPU.
#SBATCH --job-name=m3-pseudo
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
for D in rose_cnndm rose_xsum rose_samsum natural770; do
  [ -s "$M3/m31/pseudo_$D.jsonl" ] || "$PLAN_PY" "$CODE/b20_build_pseudo.py" --domain "$D" --split train --n-cands 4 --out "$M3/m31/pseudo_$D.jsonl"
done
cat "$M3"/m31/pseudo_rose_cnndm.jsonl "$M3"/m31/pseudo_rose_xsum.jsonl "$M3"/m31/pseudo_rose_samsum.jsonl "$M3"/m31/pseudo_natural770.jsonl > "$M3/m31/pseudo_all.jsonl"
wc -l "$M3/m31/pseudo_all.jsonl"
