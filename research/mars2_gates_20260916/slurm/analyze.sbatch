#!/bin/bash
# B18/B21 analysis on the ACU (human labels) and m2 (B1 labels) unit sets; CPU. Env: GATES_SPLIT.
#SBATCH --job-name=g-analyze
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/user/results/mars2_gates/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
SPLIT="${GATES_SPLIT:-validation}"; OUT="$G/analysis${GATES_TAG:+_$GATES_TAG}"; mkdir -p "$OUT"; plan_receipt "$OUT"
DUMPS=""; for ARM in full_recovery direct_enabled; do P="$(b2_dumps "$ARM" "$SPLIT")"; [ -n "$P" ] && DUMPS="$DUMPS $ARM=$P"; done
OJ=""; for M in Qwen_Qwen3.8-27B google_gemma-4-31B-it; do F="$PLAN_OUT/b3/open_judge/${M}_${SPLIT}_all.jsonl"; [ -s "$F" ] && OJ="$OJ $M=$F"; done
"$PLAN_PY" "$CODE/analyze.py" --unit-set acu --split "$SPLIT" --units "$G/b18/acu_units_${SPLIT}.jsonl" --mars2-glob "$G/b18/mars2/*_${SPLIT}.jsonl" \
  ${DUMPS:+--dumps $DUMPS} --scores-glob "$G/b21/scores/acu_*.jsonl" --judge-glob "$G/b18/judge/judge_shard*.jsonl" ${OJ:+--open-judge $OJ} --out "$OUT"
"$PLAN_PY" "$CODE/analyze.py" --unit-set m2 --split "$SPLIT" ${DUMPS:+--dumps $DUMPS} --scores-glob "$G/b21/scores/m2_*.jsonl" ${OJ:+--open-judge $OJ} ${GATES_REF:+--reference-system "$GATES_REF"} --out "$OUT"
sha256sum "$OUT"/*.json > "$OUT/SHA256SUMS"; plan_done "$OUT"
