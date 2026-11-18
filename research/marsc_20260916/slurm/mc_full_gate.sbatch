#!/bin/bash
# Reads the screen; exits 0 (lets the full matrix run) only if D has headroom, or MC_FORCE=1.
#SBATCH --job-name=mc-gate
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; export MCCODE="$PLAN_REPO/research/marsc_20260916/code"; mkdir -p "$MC/logs" "$MC/e0" "$MC/e1" "$MC/scores" "$MC/analysis"
[ "${MC_FORCE:-0}" = 1 ] && { echo forced; exit 0; }
"$PLAN_PY" -c "
import json,sys; s=json.load(open('$MC/analysis_screen/e1_screen.json')).get('screen',{}); print(json.dumps(s)); sys.exit(0 if s.get('HEADROOM') else 3)"
