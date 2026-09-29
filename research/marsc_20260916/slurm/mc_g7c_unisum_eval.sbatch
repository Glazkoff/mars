#!/bin/bash
% SUPERSEDED-BY: mc_r01_consensus_fix.sbatch
% This job selected its score files with a wildcard that matched more files than it intended (the
% pseudo-seed *_seedavg_div1.jsonl, or the importance-policy outputs *_imp0.2_div1.jsonl). It is kept
% unchanged as the record of what was actually run; its numbers are superseded. Do not re-run it to
% reproduce the paper -- run the job named above, which reads explicit three-file manifests.
# MARS-C G7c -- UniSumEval: frozen inventory (propositions + entities), frozen verifiers (E1b human-fact x3, B21 judge-label x3),
# the rule in both regimes, emitted-top-k recall of human-omitted key facts; per-domain breakdown. One GPU.
#SBATCH --job-name=mc-g7c
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"; G7="$MC/g7_unisum"
ACU="$G7/acu_units_unisum.jsonl"; INV="$G7/units_gemma+ent_unisum.jsonl"
"$PLAN_PY" "$MCCODE/mc_unisum.py" inventory --acu "$ACU" --props "$G7"/props/props_unisum_shard*.jsonl --out "$G7"
"$PLAN_PY" "$MCCODE/mc_score_units.py" --units "gemma+ent=$INV" --models "$MC/e1/mc-A-lam1-crossenc-inv_seed*/model" --tag humanfact --scores-out "$G7/scores"
"$PLAN_PY" "$M3CODE/m31_score_pipeline_all.py" --units "gemma+ent=$INV" --ckpts "$G/b21/xenc_seed*" --scores-out "$G7/scores"
div () { "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "gemma+ent=$INV" --scores-glob "$1" --out "$G7/val"; }
div "$G7/scores/gemma+ent_humanfact_seed*.jsonl"; div "$G7/scores/gemma+ent_xenc_seed*.jsonl"
for T in humanfact xenc; do "$PLAN_PY" "$MCCODE/mc_variants.py" seedavg --units "gemma+ent=$INV" --scores "$G7/scores/gemma+ent_${T}_seed*.jsonl" --tag "$T" --out "$G7/inv"; div "$G7/inv/gemma+ent_${T}_seedavg.jsonl"; done
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --acu "$ACU" --k 5 10 --n-boot 500 --primary "humanfact+div1" --out "$G7/eval_unisum" --dump-emitted "$G7/emitted_unisum.jsonl" --systems \
  "humanfact+div1=$INV:$G7/val/gemma+ent_humanfact_seed*_div1.jsonl" "judgefact+div1=$INV:$G7/val/gemma+ent_xenc_seed*_div1.jsonl" \
  "humanfact.single+div1=$INV:$G7/val/gemma+ent_humanfact_seedavg_div1.jsonl" "judgefact.single+div1=$INV:$G7/val/gemma+ent_xenc_seedavg_div1.jsonl" \
  "humanfact=$INV:$G7/scores/gemma+ent_humanfact_seed*.jsonl" "judgefact=$INV:$G7/scores/gemma+ent_xenc_seed*.jsonl" 2>&1 | grep "\[e2"
plan_done "$G7"
