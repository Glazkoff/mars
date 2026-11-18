#!/bin/bash
% SUPERSEDED-BY: mc_r09_validation_fix.sbatch
% This job selected its score files with a wildcard that matched more files than it intended (the
% pseudo-seed *_seedavg_div1.jsonl, or the importance-policy outputs *_imp0.2_div1.jsonl). It is kept
% unchanged as the record of what was actually run; its numbers are superseded. Do not re-run it to
% reproduce the paper -- run the job named above, which reads explicit three-file manifests.
# MARS-C E2: full comparison under the identical one-unit-per-sentence emission rule, no learned policy (amendment 2026-09-16 16:35 UTC). CPU.
#SBATCH --job-name=mc-e2div-full
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"; IMP2="$MC/e2imp_all"; DIV="$MC/e2div"; DF="$MC/e2div_full"
mkdir -p "$DF/val"; plan_receipt "$DF"
INV_GE="$M3/m32/units_gemma+ent_validation.jsonl"; INV_DI="$M3/m32/units_distilled_validation.jsonl"; INV_GM="$M3/m32/units_gemma_validation.jsonl"; INV_SP="$M3/m32/units_spacy_validation.jsonl"
HF_GE="$MC/scores/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*.jsonl"; HF_DI="$IMP2/scores/distilled_humanfact_seed*.jsonl"
div () { "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "$1" --scores-glob "$2" --out "$DF/val"; }
div "gemma+ent=$INV_GE" "$M3/scores/units_gemma+ent_xenc_seed*.jsonl"; div "gemma=$INV_GM" "$M3/scores/units_gemma_xenc_seed*.jsonl"
div "spacy=$INV_SP" "$M3/scores/units_spacy_xenc_seed*.jsonl"; div "spacy=$INV_SP" "$M3/scores/units_spacy_tagger_seed*.jsonl"
div "gemma+ent=$INV_GE" "$M3/scores/units_gemma+ent_tagger_facts_seed*.jsonl"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --k 5 10 --n-boot 300 --primary "humanfact+div1" --out "$DF/eval_validation" --systems \
  "humanfact+div1=$INV_GE:$DIV/val/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*_div1.jsonl" \
  "humanfact.distilled+div1=$INV_DI:$DIV/val/distilled_humanfact_seed*_div1.jsonl" \
  "judgefact+div1=$INV_GE:$DF/val/units_gemma+ent_xenc_seed*_div1.jsonl" \
  "judgefact.distilled+div1=$INV_DI:$DIV/val/units_distilled_xenc_seed*_div1.jsonl" \
  "judgefact.gemma+div1=$INV_GM:$DF/val/units_gemma_xenc_seed*_div1.jsonl" \
  "judgefact.spacy+div1=$INV_SP:$DF/val/units_spacy_xenc_seed*_div1.jsonl" \
  "tagger_facts+div1=$INV_GE:$DF/val/units_gemma+ent_tagger_facts_seed*_div1.jsonl" \
  "tagger.spacy+div1=$INV_SP:$DF/val/units_spacy_tagger_seed*_div1.jsonl" \
  "humanfact+imp+div1=$INV_GE:$DIV/val/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*_imp0.2_div1.jsonl" \
  "humanfact.distilled+imp=$INV_DI:$IMP2/val/distilled_humanfact_seed*_imp0.2.jsonl" \
  "humanfact=$INV_GE:$HF_GE" "humanfact.distilled=$INV_DI:$HF_DI" \
  "judgefact.distilled=$INV_DI:$M3/scores/units_distilled_xenc_seed*.jsonl" \
  "tagger.spacy=$INV_SP:$M3/scores/units_spacy_tagger_seed*.jsonl" \
  "mars2.spacy=dumps:$PLAN_OUT/b2/scores/direct_enabled_seed*/units_validation.jsonl" 2>&1 | grep "\[e2"
plan_done "$DF"
