#!/bin/bash
# MARS-3 environment; source after plan2026/common.sh and gates env.sh
export M3="${MARS3_OUT:-/home/user/results/mars3}"; export MARS3_OUT="$M3"
export M3CODE="$PLAN_REPO/research/mars3_20260916/code"
mkdir -p "$M3/logs" "$M3/scores" "$M3/m31" "$M3/m32" "$M3/analysis"
