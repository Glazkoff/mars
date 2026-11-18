#!/bin/bash
% SUPERSEDED-BY: mc_r09_validation_fix.sbatch
% This job selected its score files with a wildcard that matched more files than it intended (the
% pseudo-seed *_seedavg_div1.jsonl, or the importance-policy outputs *_imp0.2_div1.jsonl). It is kept
% unchanged as the record of what was actually run; its numbers are superseded. Do not re-run it to
% reproduce the paper -- run the job named above, which reads explicit three-file manifests.
# MARS-C G3b -- (a) union inventory with the corrected file globs; (b) does the frozen result depend on averaging three per-seed
# rule outputs (consensus), or does a single seed / a seed-averaged score file reproduce it? Decides how the rule is defined for TEST.
#SBATCH --job-name=mc-g3b
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"; IMP2="$MC/e2imp_all"; DIV="$MC/e2div"; G3="$MC/g3"; B="$G3/b"; mkdir -p "$B/val" "$B/inv"
INV_GE="$M3/m32/units_gemma+ent_validation.jsonl"; INV_DI="$M3/m32/units_distilled_validation.jsonl"; INV_UN="$G3/inv/units_union_validation.jsonl"
HF_GE="$MC/scores/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*.jsonl"; JF_GE="$M3/scores/units_gemma+ent_xenc_seed*.jsonl"; JF_DI="$M3/scores/units_distilled_xenc_seed*.jsonl"; ACU="$G/b18/acu_units_validation.jsonl"
div () { "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "$1" --scores-glob "$2" --out "$3"; }
# (a) union, correct globs
div "union=$INV_UN" "$G3/inv/union_*crossenc-inv_seed*.jsonl" "$B/val"; div "union=$INV_UN" "$G3/inv/union_*_xenc_seed*.jsonl" "$B/val"
# (b) seed handling: seed-averaged human-fact scores -> one file -> rule; single seed -> rule; judge-label seed-averaged -> rule
"$PLAN_PY" "$MCCODE/mc_variants.py" ensemble --units "gemma+ent=$INV_GE" --scores-a "$HF_GE" --scores-b "$HF_GE" --tag hfavg --out "$B/inv"
"$PLAN_PY" "$MCCODE/mc_variants.py" ensemble --units "distilled=$INV_DI" --scores-a "$JF_DI" --scores-b "$JF_DI" --tag jfavg --out "$B/inv"
div "gemma+ent=$INV_GE" "$B/inv/gemma+ent_hfavg_seedens.jsonl" "$B/val"; div "distilled=$INV_DI" "$B/inv/distilled_jfavg_seedens.jsonl" "$B/val"
div "gemma+ent=$INV_GE" "$MC/scores/units_gemma+ent_mc-A-lam1-crossenc-inv_seed20260906.jsonl" "$B/val"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --acu "$ACU" --k 10 --n-boot 300 --primary "humanfact+div1" --out "$B/eval_validation" --systems \
  "humanfact+div1=$INV_GE:$DIV/val/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*_div1.jsonl" \
  "humanfact.seedavg+div1=$INV_GE:$B/val/gemma+ent_hfavg_seedens_div1.jsonl" \
  "humanfact.seed06+div1=$INV_GE:$B/val/units_gemma+ent_mc-A-lam1-crossenc-inv_seed20260906_div1.jsonl" \
  "judgefact.distilled+div1=$INV_DI:$DIV/val/units_distilled_xenc_seed*_div1.jsonl" \
  "judgefact.distilled.seedavg+div1=$INV_DI:$B/val/distilled_jfavg_seedens_div1.jsonl" \
  "union_humanfact+div1=$INV_UN:$B/val/union_gemma+ent_mc-A-lam1-crossenc-inv_seed*_div1.jsonl" \
  "union_judgefact+div1=$INV_UN:$B/val/union_gemma+ent_xenc_seed*_div1.jsonl" \
  "ensemble+div1=$INV_GE:$G3/val/gemma+ent_ens_seedens_div1.jsonl" 2>&1 | grep "\[e2"
