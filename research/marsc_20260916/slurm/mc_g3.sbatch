#!/bin/bash
% SUPERSEDED-BY: mc_r07a_crossed_build.sbatch (validation emissions) and mc_r01_consensus_fix.sbatch
% This job selected its score files with a wildcard that matched more files than it intended (the
% pseudo-seed *_seedavg_div1.jsonl, or the importance-policy outputs *_imp0.2_div1.jsonl). It is kept
% unchanged as the record of what was actually run; its numbers are superseded. Do not re-run it to
% reproduce the paper -- run the job named above, which reads explicit three-file manifests.
# MARS-C G3 -- margin variants before TEST (PLAN_MAIN_CONTRIBUTION.md): union inventory, verifier ensemble, cap 2, k 20, MARS-2 under
# the rule; then the emitted-unit dump for G2/G6, the G2 judge tasks and the G6 blinded human sample. CPU only.
#SBATCH --job-name=mc-g3
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"; IMP2="$MC/e2imp_all"; DIV="$MC/e2div"; G3="$MC/g3"
mkdir -p "$G3/val" "$G3/inv"; plan_receipt "$G3"
INV_GE="$M3/m32/units_gemma+ent_validation.jsonl"; INV_DI="$M3/m32/units_distilled_validation.jsonl"; INV_SP="$M3/m32/units_spacy_validation.jsonl"
HF_GE="$MC/scores/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*.jsonl"; HF_DI="$IMP2/scores/distilled_humanfact_seed*.jsonl"
JF_GE="$M3/scores/units_gemma+ent_xenc_seed*.jsonl"; JF_DI="$M3/scores/units_distilled_xenc_seed*.jsonl"
ACU="$G/b18/acu_units_validation.jsonl"
# 1. variants
"$PLAN_PY" "$MCCODE/mc_variants.py" union --units "gemma+ent=$INV_GE" "distilled=$INV_DI" --scores-a "$HF_GE" "$JF_GE" --scores-b "$HF_DI" "$JF_DI" --name union --split validation --out "$G3/inv"
"$PLAN_PY" "$MCCODE/mc_variants.py" ensemble --units "gemma+ent=$INV_GE" --scores-a "$HF_GE" --scores-b "$JF_GE" --tag ens --out "$G3/inv"
"$PLAN_PY" "$MCCODE/mc_variants.py" ensemble --units "distilled=$INV_DI" --scores-a "$HF_DI" --scores-b "$JF_DI" --tag ens --out "$G3/inv"
"$PLAN_PY" "$MCCODE/mc_variants.py" dumps2units --dumps "$PLAN_OUT/b2/scores/direct_enabled_seed*/units_validation.jsonl" --source-units "$INV_SP" --name mars2 --split validation --out "$G3/inv"
INV_UN="$G3/inv/units_union_validation.jsonl"; INV_M2="$G3/inv/units_mars2_validation.jsonl"
# 2. the rule (cap 1) on every variant; cap 2 on the frozen pipeline
div () { "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "$1" --scores-glob "$2" --out "$G3/val" ${3:+--cap $3}; }
div "union=$INV_UN" "$G3/inv/union_mc-A-lam1-crossenc-inv_seed*.jsonl"; div "union=$INV_UN" "$G3/inv/union_xenc_seed*.jsonl"
div "gemma+ent=$INV_GE" "$G3/inv/gemma+ent_ens_seedens.jsonl"; div "distilled=$INV_DI" "$G3/inv/distilled_ens_seedens.jsonl"
div "mars2=$INV_M2" "$G3/inv/mars2_mars2_seed0.jsonl"; div "gemma+ent=$INV_GE" "$HF_GE" 2
# 3. evaluate everything against the frozen pipeline; dump emitted units for G2 / G6
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --acu "$ACU" --k 5 10 20 --n-boot 300 --primary "humanfact+div1" --out "$G3/eval_validation" --dump-emitted "$G3/emitted_validation.jsonl" --systems \
  "humanfact+div1=$INV_GE:$DIV/val/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*_div1.jsonl" \
  "union_humanfact+div1=$INV_UN:$G3/val/union_mc-A-lam1-crossenc-inv_seed*_div1.jsonl" \
  "union_judgefact+div1=$INV_UN:$G3/val/union_xenc_seed*_div1.jsonl" \
  "ensemble+div1=$INV_GE:$G3/val/gemma+ent_ens_seedens_div1.jsonl" \
  "ensemble.distilled+div1=$INV_DI:$G3/val/distilled_ens_seedens_div1.jsonl" \
  "humanfact+div2=$INV_GE:$G3/val/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*_div2.jsonl" \
  "humanfact.distilled+div1=$INV_DI:$DIV/val/distilled_humanfact_seed*_div1.jsonl" \
  "judgefact.distilled+div1=$INV_DI:$DIV/val/units_distilled_xenc_seed*_div1.jsonl" \
  "tagger.spacy+div1=$INV_SP:$MC/e2div_full/val/units_spacy_tagger_seed*_div1.jsonl" \
  "mars2+div1=$INV_M2:$G3/val/mars2_mars2_seed0_div1.jsonl" \
  "mars2=$INV_M2:$G3/inv/mars2_mars2_seed0.jsonl" \
  "humanfact=$INV_GE:$HF_GE" "judgefact.distilled=$INV_DI:$JF_DI" 2>&1 | grep "\[e2"
# 4. G2 judge tasks (five emitters) and G6 blinded human sample (three emitters)
"$PLAN_PY" "$MCCODE/mc_judge_precision.py" build --emitted "$G3/emitted_validation.jsonl" --acu "$ACU" --systems humanfact+div1 humanfact.distilled+div1 judgefact.distilled+div1 tagger.spacy+div1 mars2 --out "$MC/g2"
"$PLAN_PY" "$MCCODE/mc_human_sample.py" --emitted "$G3/emitted_validation.jsonl" --acu "$ACU" --systems humanfact+div1 judgefact.distilled+div1 mars2 --per-system 100 --out "$MC/g6"
plan_done "$G3"
