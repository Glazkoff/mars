#!/bin/bash
# Environment for the MARS-2 gates (B18-B22). Source after slurm/cluster-B/plan2026/common.sh.
export G="${MARS2_GATES_OUT:-/home/user/results/mars2_gates}"
export MARS2_GATES_OUT="$G"
export CODE="$PLAN_REPO/research/mars2_gates_20260916/code"
export B1_DIR="${B1_DIR:-/home/user/data/plan2026/b1}"
export MODELS_B2="/home/user/models/plan2026/b2"
export MARS2_DEVICE="${MARS2_DEVICE:-cuda}" MARS2_SPACY_MODEL=en_core_web_sm
mkdir -p "$G/logs" "$G/b18" "$G/b20" "$G/b21/scores" "$G/analysis"
snap() { ls -d "$HF_HOME/hub/models--${1//\//--}/snapshots"/*/ 2>/dev/null | head -1; }
b2_dumps() { { ls $PLAN_OUT/b2/scores/${1}_seed*/units_${2:-validation}.jsonl 2>/dev/null || true; } | paste -sd, -; }   # glob unquoted on purpose
