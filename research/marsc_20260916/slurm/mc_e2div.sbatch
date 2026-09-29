#!/bin/bash
% SUPERSEDED-BY: mc_r09_validation_fix.sbatch
% This job selected its score files with a wildcard that matched more files than it intended (the
% pseudo-seed *_seedavg_div1.jsonl, or the importance-policy outputs *_imp0.2_div1.jsonl). It is kept
% unchanged as the record of what was actually run; its numbers are superseded. Do not re-run it to
% reproduce the paper -- run the job named above, which reads explicit three-file manifests.
# MARS-C E2: (a) exact paired CIs of the policy controls on the gemma+ent inventory (primary = human-fact + learned policy over
# gemma+ent); (b) the registered one-unit-per-sentence diversity step applied identically to every emitter (amendment
# 2026-09-16 16:10 UTC). CPU only.
#SBATCH --job-name=mc-e2div
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"; IMP2="$MC/e2imp_all"; CTL="$MC/e2ctl"; DIV="$MC/e2div"
mkdir -p "$DIV/val"; plan_receipt "$DIV"
INV_GE="$M3/m32/units_gemma+ent_validation.jsonl"; INV_DI="$M3/m32/units_distilled_validation.jsonl"
TAUL=$(cat "$CTL/chosen_tau_leadpos.txt"); TR_GE=$("$PLAN_PY" -c "import json;print(json.load(open('$CTL/scores/controls_manifest.json'))['gemma+ent']['tau_random'])")
HF_GE="$MC/scores/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*.jsonl"; HF_DI="$IMP2/scores/distilled_humanfact_seed*.jsonl"
# (a) exact controls on gemma+ent
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --k 5 10 --n-boot 300 --primary "humanfact+imp" --out "$DIV/eval_controls_gemma+ent" --systems \
  "humanfact+imp=$INV_GE:$IMP2/val/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*_imp0.2.jsonl" \
  "humanfact+leadpos=$INV_GE:$CTL/val_lead/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*_imp$TAUL.jsonl" \
  "humanfact+random=$INV_GE:$CTL/val_rand/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*_imp$TR_GE.jsonl" \
  "judgefact+imp=$INV_GE:$IMP2/val/units_gemma+ent_xenc_seed*_imp0.2.jsonl" \
  "humanfact=$INV_GE:$HF_GE" 2>&1 | grep "\[e2 k"
# (b) diversity step on the policy-filtered and the unfiltered emitters
div () { "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "$1" --scores-glob "$2" --out "$DIV/val"; }
div "distilled=$INV_DI" "$IMP2/val/distilled_humanfact_seed*_imp0.2.jsonl"; div "distilled=$INV_DI" "$IMP2/val/units_distilled_xenc_seed*_imp0.2.jsonl"
div "gemma+ent=$INV_GE" "$IMP2/val/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*_imp0.2.jsonl"; div "gemma+ent=$INV_GE" "$IMP2/val/units_gemma+ent_xenc_seed*_imp0.2.jsonl"
div "distilled=$INV_DI" "$HF_DI"; div "gemma+ent=$INV_GE" "$HF_GE"; div "distilled=$INV_DI" "$M3/scores/units_distilled_xenc_seed*.jsonl"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --k 5 10 --n-boot 300 --primary "humanfact.distilled+imp+div1" --out "$DIV/eval_validation" --systems \
  "humanfact.distilled+imp+div1=$INV_DI:$DIV/val/distilled_humanfact_seed*_imp0.2_div1.jsonl" \
  "humanfact.distilled+imp=$INV_DI:$IMP2/val/distilled_humanfact_seed*_imp0.2.jsonl" \
  "judgefact.distilled+imp+div1=$INV_DI:$DIV/val/units_distilled_xenc_seed*_imp0.2_div1.jsonl" \
  "judgefact.distilled+imp=$INV_DI:$IMP2/val/units_distilled_xenc_seed*_imp0.2.jsonl" \
  "humanfact+imp+div1=$INV_GE:$DIV/val/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*_imp0.2_div1.jsonl" \
  "humanfact+imp=$INV_GE:$IMP2/val/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*_imp0.2.jsonl" \
  "judgefact+imp+div1=$INV_GE:$DIV/val/units_gemma+ent_xenc_seed*_imp0.2_div1.jsonl" \
  "humanfact.distilled+div1=$INV_DI:$DIV/val/distilled_humanfact_seed*_div1.jsonl" \
  "humanfact.distilled=$INV_DI:$HF_DI" \
  "humanfact+div1=$INV_GE:$DIV/val/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*_div1.jsonl" \
  "judgefact.distilled+div1=$INV_DI:$DIV/val/units_distilled_xenc_seed*_div1.jsonl" \
  "judgefact.distilled=$INV_DI:$M3/scores/units_distilled_xenc_seed*.jsonl" 2>&1 | grep "\[e2"
plan_done "$DIV"
