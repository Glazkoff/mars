#!/bin/bash
# MARS-C G7e -- rescore the 1,822 UniSumEval pairs: six tasks = {human-fact, judge-label} x 3 seeds, one GPU each.
#SBATCH --job-name=mc-g7e
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-5
#SBATCH --output=/home/user/results/marsc/logs/%x-%A_%a.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"; G7="$MC/g7_unisum"; INV="$G7/units_gemma+ent_unisum.jsonl"
T="${SLURM_ARRAY_TASK_ID:-0}"; SEEDS=(20260906 20260907 20260908); S="${SEEDS[$((T % 3))]}"
if [ $((T / 3)) -eq 0 ]; then
  "$PLAN_PY" "$MCCODE/mc_score_units.py" --units "gemma+ent=$INV" --models "$MC/e1/mc-A-lam1-crossenc-inv_seed$S/model" --tag humanfact --scores-out "$G7/scores"
else
  "$PLAN_PY" "$M3CODE/m31_score_pipeline_all.py" --units "gemma+ent=$INV" --ckpts "$G/b21/xenc_seed$S" --scores-out "$G7/scores"
fi
