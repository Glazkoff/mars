#!/bin/bash
% SUPERSEDED-BY: mc_r09b_policy_cells.sbatch
% This job selected its score files with a wildcard that matched more files than it intended (the
% pseudo-seed *_seedavg_div1.jsonl, or the importance-policy outputs *_imp0.2_div1.jsonl). It is kept
% unchanged as the record of what was actually run; its numbers are superseded. Do not re-run it to
% reproduce the paper -- run the job named above, which reads explicit three-file manifests.
# MARS-C E2 with the frozen source-only importance policy (registered amendment). One GPU job:
#  build rows -> train importance tagger (3 seeds) -> importance scores on tune + validation inventories -> omission scores on the tune
#  inventory (human-fact + judge-label verifiers) -> choose tau on TUNE (emitted recall@10) -> combine on validation -> E2 eval.
#SBATCH --job-name=mc-e2imp
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"; IMP="$MC/e2imp"; mkdir -p "$IMP/scores" "$IMP/tune" "$IMP/val"; plan_receipt "$IMP"
INVV="$M3/m32/units_gemma+ent_validation.jsonl"
# 1. rows + tune units
MARS2_DEVICE=cpu "$PLAN_PY" "$MCCODE/mc_importance.py" build --data-dir "$MC/e0" --inventory-rows "$M3/m32/train_rows_facts.jsonl" --inventory-name gemma+ent --out "$IMP"
INVT="$IMP/units_gemma+ent_tune.jsonl"
# 2. importance tagger (source-only) x 3 seeds, scoring both inventories
for SEED in 20260906 20260907 20260908; do
  "$PLAN_PY" "$M3CODE/m31_tagger.py" --tag importance --seed "$SEED" --out "$IMP/tagger_seed$SEED" --train "$IMP/imp_rows_train.jsonl" --source-only --epochs 3 \
     --eval-m2 /dev/null --eval-acu /nonexistent --eval-units "gemma+ent=$INVV" "gemma+ent_tune=$INVT" --scores-out "$IMP/scores" 2>&1 | grep -v Warning | tail -3
done
# 3. omission scores on the TUNE inventory: human-fact verifier (saved E2-pilot models) and judge-label cross-encoders
"$PLAN_PY" "$MCCODE/mc_score_units.py" --units "gemma+ent_tune=$INVT" --models "$MC/e1/mc-A-lam1-crossenc-inv_seed*/model" --tag humanfact --scores-out "$IMP/scores"
"$PLAN_PY" "$M3CODE/m31_score_pipeline_all.py" --units "gemma+ent_tune=$INVT" --scores-out "$IMP/scores"
# 4. choose tau on TUNE: filtered human-fact emitter, emitted recall@10 of human omissions
for TAU in 0.2 0.35 0.5 0.65 0.8; do
  "$PLAN_PY" "$MCCODE/mc_importance.py" combine --omission-glob "$IMP/scores/gemma+ent_tune_humanfact_seed*.jsonl" --importance-glob "$IMP/scores/gemma+ent_tune_importance_seed*.jsonl" --tau $TAU --out "$IMP/tune"
  "$PLAN_PY" "$MCCODE/mc_e2_eval.py" --acu "$IMP/acu_units_tune.jsonl" --k 10 --n-boot 50 --out "$IMP/tune/eval_tau$TAU" --systems "hf_imp$TAU=$INVT:$IMP/tune/gemma+ent_tune_humanfact_seed*_imp$TAU.jsonl" "hf_nofilter=$INVT:$IMP/scores/gemma+ent_tune_humanfact_seed*.jsonl" 2>&1 | grep "\[e2 k"
done
TAU=$("$PLAN_PY" - <<PY
import json,glob,re
best=None
for f in glob.glob("$IMP/tune/eval_tau*/e2_emitted.json"):
    tau=float(re.search(r"tau([0-9.]+)",f).group(1)); r=json.load(open(f))["k"]["10"]["systems"]; rec=[v["recall"] for k,v in r.items() if k.startswith("hf_imp")][0]
    if best is None or rec>best[1]: best=(tau,rec)
print(best[0])
PY
); echo "[e2imp] chosen tau on TUNE = $TAU"
# 5. validation: filter every emitter identically, evaluate
for G in "$MC/scores/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*.jsonl" "$M3/scores/units_gemma+ent_xenc_seed*.jsonl" "$M3/scores/units_gemma+ent_tagger_facts_seed*.jsonl"; do
  "$PLAN_PY" "$MCCODE/mc_importance.py" combine --omission-glob "$G" --importance-glob "$IMP/scores/gemma+ent_importance_seed*.jsonl" --tau $TAU --out "$IMP/val"
done
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --k 5 10 --n-boot 300 --primary "humanfact+imp" --out "$IMP/eval_validation" --systems \
  "humanfact+imp=$INVV:$IMP/val/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*_imp$TAU.jsonl" "humanfact=$INVV:$MC/scores/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*.jsonl" \
  "judgefact+imp=$INVV:$IMP/val/units_gemma+ent_xenc_seed*_imp$TAU.jsonl" "judgefact=$INVV:$M3/scores/units_gemma+ent_xenc_seed*.jsonl" \
  "judgefact.distilled=$M3/m32/units_distilled_validation.jsonl:$M3/scores/units_distilled_xenc_seed*.jsonl" \
  "tagger_facts+imp=$INVV:$IMP/val/units_gemma+ent_tagger_facts_seed*_imp$TAU.jsonl" "tagger.spacy=$M3/m32/units_spacy_validation.jsonl:$M3/scores/units_spacy_tagger_seed*.jsonl" \
  "mars2.spacy=dumps:$PLAN_OUT/b2/scores/direct_enabled_seed*/units_validation.jsonl" 2>&1 | grep "\[e2"
echo "$TAU" > "$IMP/chosen_tau.txt"; plan_done "$IMP"
