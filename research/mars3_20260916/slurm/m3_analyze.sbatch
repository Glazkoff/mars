#!/bin/bash
# MARS-3 analysis: acu (inventory-free + pipeline-aligned) and m2, with the cross-encoder as reference. Env: GATES_TAG.
#SBATCH --job-name=m3-analyze
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/results/mars3/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
OUT="$M3/analysis${GATES_TAG:+_$GATES_TAG}"; mkdir -p "$OUT"; plan_receipt "$OUT"
DUMPS=""; for ARM in full_recovery direct_enabled; do P="$(b2_dumps "$ARM" validation)"; [ -n "$P" ] && DUMPS="$DUMPS $ARM=$P"; done
OJ=""; for M in Qwen_Qwen3.8-27B google_gemma-4-31B-it; do F="$PLAN_OUT/b3/open_judge/${M}_validation_all.jsonl"; [ -s "$F" ] && OJ="$OJ $M=$F"; done
AL=""; for F in "$M3"/m32/units_*_validation.jsonl; do N=$(basename "$F" _validation.jsonl); ls "$M3"/scores/${N}_xenc_seed*.jsonl >/dev/null 2>&1 && AL="$AL $N=$F:$M3/scores/${N}_xenc_seed*.jsonl"; done
"$PLAN_PY" "$CODE/analyze.py" --unit-set acu --units "$G/b18/acu_units_validation.jsonl" --mars2-glob "$G/b18/mars2/*_validation.jsonl" ${DUMPS:+--dumps $DUMPS} \
  --scores-glob "$G/b21/scores/acu_*.jsonl" "$M3/scores/acu_*.jsonl" --judge-glob "$G/b18/judge/judge_shard*.jsonl" ${OJ:+--open-judge $OJ} ${AL:+--aligned-pipeline $AL} \
  --primary "m3:tagger" --reference-system "xenc:deberta-large-mnli" --out "$OUT"
"$PLAN_PY" "$CODE/analyze.py" --unit-set m2 ${DUMPS:+--dumps $DUMPS} --scores-glob "$G/b21/scores/m2_*.jsonl" "$M3/scores/m2_*.jsonl" ${OJ:+--open-judge $OJ} \
  --primary "m3:tagger" --reference-system "xenc:deberta-large-mnli" --out "$OUT"
sha256sum "$OUT"/*.json > "$OUT/SHA256SUMS"; plan_done "$OUT"
