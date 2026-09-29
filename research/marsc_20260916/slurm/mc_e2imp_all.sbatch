#!/bin/bash
% SUPERSEDED-BY: mc_r09b_policy_cells.sbatch
% This job selected its score files with a wildcard that matched more files than it intended (the
% pseudo-seed *_seedavg_div1.jsonl, or the importance-policy outputs *_imp0.2_div1.jsonl). It is kept
% unchanged as the record of what was actually run; its numbers are superseded. Do not re-run it to
% reproduce the paper -- run the job named above, which reads explicit three-file manifests.
# MARS-C E2 importance policy, completeness run (registered amendment 2026-09-16 15:10 UTC): the identical frozen policy
# (tau fixed from 4835's TUNE choice) applied to EVERY inventory the judge-label pipelines use, so the bar's comparator is the
# strongest judge-label pipeline under the same policy. Importance taggers retrained with the same seeds/config (saved).
#SBATCH --job-name=mc-e2imp-all
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"; IMP="$MC/e2imp"; IMP2="$MC/e2imp_all"
mkdir -p "$IMP2/scores" "$IMP2/val"; plan_receipt "$IMP2"
TAU=$(cat "$IMP/chosen_tau.txt"); echo "[e2imp-all] tau fixed from TUNE = $TAU"
INV_GE="$M3/m32/units_gemma+ent_validation.jsonl"; INV_DI="$M3/m32/units_distilled_validation.jsonl"; INV_GM="$M3/m32/units_gemma_validation.jsonl"; INV_SP="$M3/m32/units_spacy_validation.jsonl"
# 1. importance tagger (source-only, same rows/seeds/config as 4835), scored on all four validation inventories, models saved
for SEED in 20260906 20260907 20260908; do
  "$PLAN_PY" "$M3CODE/m31_tagger.py" --tag importance_all --seed "$SEED" --out "$IMP2/tagger_seed$SEED" --train "$IMP/imp_rows_train.jsonl" --source-only --epochs 3 --save-model \
     --eval-m2 /dev/null --eval-acu /nonexistent --eval-units "gemma+ent=$INV_GE" "distilled=$INV_DI" "gemma=$INV_GM" "spacy=$INV_SP" --scores-out "$IMP2/scores" 2>&1 | grep -v Warning | tail -5
done
# 2. human-fact verifier omission scores on the distilled inventory (saved E1b/E2-pilot models); gemma+ent scores exist from the pilot
"$PLAN_PY" "$MCCODE/mc_score_units.py" --units "distilled=$INV_DI" --models "$MC/e1/mc-A-lam1-crossenc-inv_seed*/model" --tag humanfact --scores-out "$IMP2/scores"
# 3. filter every emitter identically (importance averaged over the 3 seeds of its inventory)
comb () { "$PLAN_PY" "$MCCODE/mc_importance.py" combine --omission-glob "$1" --importance-glob "$IMP2/scores/$2_importance_all_seed*.jsonl" --tau "$TAU" --out "$IMP2/val"; }
comb "$MC/scores/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*.jsonl" "gemma+ent"
comb "$IMP2/scores/distilled_humanfact_seed*.jsonl" "distilled"
comb "$M3/scores/units_gemma+ent_xenc_seed*.jsonl" "gemma+ent"
comb "$M3/scores/units_distilled_xenc_seed*.jsonl" "distilled"
comb "$M3/scores/units_gemma_xenc_seed*.jsonl" "gemma"
comb "$M3/scores/units_spacy_xenc_seed*.jsonl" "spacy"
comb "$M3/scores/units_gemma+ent_tagger_facts_seed*.jsonl" "gemma+ent"
comb "$M3/scores/units_spacy_tagger_seed*.jsonl" "spacy"
# 4. evaluate: emitted top-k recall of human-omitted ACUs, paired document bootstrap against the primary
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --k 5 10 --n-boot 300 --primary "humanfact+imp" --out "$IMP2/eval_validation" --systems \
  "humanfact+imp=$INV_GE:$IMP2/val/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*_imp$TAU.jsonl" \
  "humanfact.distilled+imp=$INV_DI:$IMP2/val/distilled_humanfact_seed*_imp$TAU.jsonl" \
  "judgefact+imp=$INV_GE:$IMP2/val/units_gemma+ent_xenc_seed*_imp$TAU.jsonl" \
  "judgefact.distilled+imp=$INV_DI:$IMP2/val/units_distilled_xenc_seed*_imp$TAU.jsonl" \
  "judgefact.gemma+imp=$INV_GM:$IMP2/val/units_gemma_xenc_seed*_imp$TAU.jsonl" \
  "judgefact.spacy+imp=$INV_SP:$IMP2/val/units_spacy_xenc_seed*_imp$TAU.jsonl" \
  "tagger_facts+imp=$INV_GE:$IMP2/val/units_gemma+ent_tagger_facts_seed*_imp$TAU.jsonl" \
  "tagger.spacy+imp=$INV_SP:$IMP2/val/units_spacy_tagger_seed*_imp$TAU.jsonl" \
  "humanfact=$INV_GE:$MC/scores/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*.jsonl" \
  "humanfact.distilled=$INV_DI:$IMP2/scores/distilled_humanfact_seed*.jsonl" \
  "judgefact.distilled=$INV_DI:$M3/scores/units_distilled_xenc_seed*.jsonl" \
  "mars2.spacy=dumps:$PLAN_OUT/b2/scores/direct_enabled_seed*/units_validation.jsonl" 2>&1 | grep "\[e2"
plan_done "$IMP2"
