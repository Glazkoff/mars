#!/bin/bash
% SUPERSEDED-BY: mc_r09b_policy_cells.sbatch
% This job selected its score files with a wildcard that matched more files than it intended (the
% pseudo-seed *_seedavg_div1.jsonl, or the importance-policy outputs *_imp0.2_div1.jsonl). It is kept
% unchanged as the record of what was actually run; its numbers are superseded. Do not re-run it to
% reproduce the paper -- run the job named above, which reads explicit three-file manifests.
# MARS-C E2 policy controls (registered amendment 2026-09-16 15:45 UTC). Does the learned reference-worthiness policy beat two
# summary-blind stand-ins on the same inventories and verifiers: the lead-position prior (threshold chosen on TUNE exactly like
# the learned policy) and a random filter at the learned policy's keep rate? CPU only: every omission score already exists.
#SBATCH --job-name=mc-e2ctl
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"; IMP="$MC/e2imp"; IMP2="$MC/e2imp_all"; CTL="$MC/e2ctl"
mkdir -p "$CTL/scores" "$CTL/tune" "$CTL/val_lead" "$CTL/val_rand"; plan_receipt "$CTL"
INV_GE="$M3/m32/units_gemma+ent_validation.jsonl"; INV_DI="$M3/m32/units_distilled_validation.jsonl"; INVT="$IMP/units_gemma+ent_tune.jsonl"
# 1. heuristic importance files (+ matched keep rates of the learned policy at tau 0.2)
"$PLAN_PY" "$MCCODE/mc_policy_controls.py" --units "gemma+ent=$INV_GE" "distilled=$INV_DI" "gemma+ent_tune=$INVT" \
  --learned-glob "gemma+ent=$IMP2/scores/gemma+ent_importance_all_seed*.jsonl" "distilled=$IMP2/scores/distilled_importance_all_seed*.jsonl" --tau-learned 0.2 --out "$CTL/scores"
# 2. lead-position threshold on TUNE (human-fact emitter, recall@10), the learned policy's grid
for TAU in 0.2 0.35 0.5 0.65 0.8; do
  "$PLAN_PY" "$MCCODE/mc_importance.py" combine --omission-glob "$IMP/scores/gemma+ent_tune_humanfact_seed*.jsonl" --importance-glob "$CTL/scores/gemma+ent_tune_leadpos.jsonl" --tau $TAU --out "$CTL/tune"
  "$PLAN_PY" "$MCCODE/mc_e2_eval.py" --acu "$IMP/acu_units_tune.jsonl" --k 10 --n-boot 50 --out "$CTL/tune/eval_tau$TAU" --systems "lead_imp$TAU=$INVT:$CTL/tune/gemma+ent_tune_humanfact_seed*_imp$TAU.jsonl" 2>&1 | grep "\[e2 k"
done
TAUL=$("$PLAN_PY" - <<PY
import json,glob,re
best=None
for f in glob.glob("$CTL/tune/eval_tau*/e2_emitted.json"):
    tau=float(re.search(r"tau([0-9.]+)",f).group(1)); r=json.load(open(f))["k"]["10"]["systems"]; rec=[v["recall"] for k,v in r.items() if k.startswith("lead_imp")][0]
    if best is None or rec>best[1]: best=(tau,rec)
print(best[0])
PY
); echo "[e2ctl] lead-position tau on TUNE = $TAUL"
TR_GE=$("$PLAN_PY" -c "import json;print(json.load(open('$CTL/scores/controls_manifest.json'))['gemma+ent']['tau_random'])")
TR_DI=$("$PLAN_PY" -c "import json;print(json.load(open('$CTL/scores/controls_manifest.json'))['distilled']['tau_random'])")
echo "[e2ctl] random tau (1 - learned keep rate): gemma+ent $TR_GE, distilled $TR_DI"
# 3. apply the controls identically to the human-fact and judge-label emitters
comb () { "$PLAN_PY" "$MCCODE/mc_importance.py" combine --omission-glob "$1" --importance-glob "$2" --tau "$3" --out "$4"; }
HF_GE="$MC/scores/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*.jsonl"; HF_DI="$IMP2/scores/distilled_humanfact_seed*.jsonl"; JF_DI="$M3/scores/units_distilled_xenc_seed*.jsonl"; JF_GE="$M3/scores/units_gemma+ent_xenc_seed*.jsonl"
comb "$HF_GE" "$CTL/scores/gemma+ent_leadpos.jsonl" "$TAUL" "$CTL/val_lead"; comb "$HF_GE" "$CTL/scores/gemma+ent_random.jsonl" "$TR_GE" "$CTL/val_rand"
comb "$HF_DI" "$CTL/scores/distilled_leadpos.jsonl" "$TAUL" "$CTL/val_lead"; comb "$HF_DI" "$CTL/scores/distilled_random.jsonl" "$TR_DI" "$CTL/val_rand"
comb "$JF_DI" "$CTL/scores/distilled_leadpos.jsonl" "$TAUL" "$CTL/val_lead"; comb "$JF_DI" "$CTL/scores/distilled_random.jsonl" "$TR_DI" "$CTL/val_rand"
comb "$JF_GE" "$CTL/scores/gemma+ent_leadpos.jsonl" "$TAUL" "$CTL/val_lead"; comb "$JF_GE" "$CTL/scores/gemma+ent_random.jsonl" "$TR_GE" "$CTL/val_rand"
# 4. evaluate; primary = the learned policy on the strongest inventory (human-fact over distilled), paired against every control
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --k 5 10 --n-boot 300 --primary "humanfact.distilled+imp" --out "$CTL/eval_validation" --systems \
  "humanfact.distilled+imp=$INV_DI:$IMP2/val/distilled_humanfact_seed*_imp0.2.jsonl" \
  "humanfact.distilled+leadpos=$INV_DI:$CTL/val_lead/distilled_humanfact_seed*_imp$TAUL.jsonl" \
  "humanfact.distilled+random=$INV_DI:$CTL/val_rand/distilled_humanfact_seed*_imp$TR_DI.jsonl" \
  "humanfact.distilled=$INV_DI:$HF_DI" \
  "judgefact.distilled+imp=$INV_DI:$IMP2/val/units_distilled_xenc_seed*_imp0.2.jsonl" \
  "judgefact.distilled+leadpos=$INV_DI:$CTL/val_lead/units_distilled_xenc_seed*_imp$TAUL.jsonl" \
  "judgefact.distilled+random=$INV_DI:$CTL/val_rand/units_distilled_xenc_seed*_imp$TR_DI.jsonl" \
  "humanfact+imp=$INV_GE:$IMP2/val/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*_imp0.2.jsonl" \
  "humanfact+leadpos=$INV_GE:$CTL/val_lead/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*_imp$TAUL.jsonl" \
  "humanfact+random=$INV_GE:$CTL/val_rand/units_gemma+ent_mc-A-lam1-crossenc-inv_seed*_imp$TR_GE.jsonl" \
  "judgefact+imp=$INV_GE:$IMP2/val/units_gemma+ent_xenc_seed*_imp0.2.jsonl" \
  "judgefact+leadpos=$INV_GE:$CTL/val_lead/units_gemma+ent_xenc_seed*_imp$TAUL.jsonl" 2>&1 | grep "\[e2"
echo "$TAUL" > "$CTL/chosen_tau_leadpos.txt"; plan_done "$CTL"
