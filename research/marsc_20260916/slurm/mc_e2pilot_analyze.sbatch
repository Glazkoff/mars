#!/bin/bash
% SUPERSEDED-BY: mc_r09_validation_fix.sbatch
% This job selected its score files with a wildcard that matched more files than it intended (the
% pseudo-seed *_seedavg_div1.jsonl, or the importance-policy outputs *_imp0.2_div1.jsonl). It is kept
% unchanged as the record of what was actually run; its numbers are superseded. Do not re-run it to
% reproduce the paper -- run the job named above, which reads explicit three-file manifests.
# E2 pilot analysis: human ACUs with the pipeline-aligned human-fact verifier over the gemma+ent inventory, next to the judge-label pipelines and the taggers.
#SBATCH --job-name=mc-e2an
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; OUT="$MC/analysis_e2pilot"; mkdir -p "$OUT"; plan_receipt "$OUT"
AL="humanfact_xenc=$M3/m32/units_gemma+ent_validation.jsonl:$MC/scores/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*.jsonl"
for F in "$M3"/m32/units_*_validation.jsonl; do N=$(basename "$F" _validation.jsonl); ls "$M3"/scores/${N}_xenc_seed*.jsonl >/dev/null 2>&1 && AL="$AL judgefact_${N}=$F:$M3/scores/${N}_xenc_seed*.jsonl"; done
"$PLAN_PY" "$CODE/analyze.py" --unit-set acu --units "$G/b18/acu_units_validation.jsonl" --scores-glob "$G/b21/scores/acu_xenc_seed*.jsonl" "$M3/scores/acu_tagger_seed*.jsonl" "$M3/scores/acu_tagger_facts_seed*.jsonl" "$MC/scores/acu_mc-A-lam1-crossenc-full_seed*.jsonl" \
  --aligned-pipeline $AL --primary "pipeline-aligned:humanfact_xenc" --n-boot 500 --out "$OUT"
plan_done "$OUT"
