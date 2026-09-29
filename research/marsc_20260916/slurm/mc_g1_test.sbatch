#!/bin/bash
% SUPERSEDED-BY: mc_r01_consensus_fix.sbatch
% This job selected its score files with a wildcard that matched more files than it intended (the
% pseudo-seed *_seedavg_div1.jsonl, or the importance-policy outputs *_imp0.2_div1.jsonl). It is kept
% unchanged as the record of what was actually run; its numbers are superseded. Do not re-run it to
% reproduce the paper -- run the job named above, which reads explicit three-file manifests.
# MARS-C G1 -- TEST, one shot (PLAN_MAIN_CONTRIBUTION.md). Refuses to run unless B2_SPLIT=test is set by the user's explicit decision.
# Frozen systems: pipeline of record (G1_PRIMARY, default humanfact+div1 over gemma+ent), distilled human-fact + rule, judge-label + rule
# over gemma+ent and distilled, unruled references. Verifiers: E1b arm A (3 seeds), B21 cross-encoder (3 seeds). k 5/10, 500 bootstraps.
#SBATCH --job-name=mc-g1-test
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
[ "${B2_SPLIT:-}" = "test" ] || { echo "[g1] REFUSED: B2_SPLIT=test not set (sealed TEST split; user decision required)"; exit 3; }
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"; G1="$MC/g1_test"; mkdir -p "$G1/scores" "$G1/val" "$G1/inv"; plan_receipt "$G1"
PRIMARY="${G1_PRIMARY:-humanfact+div1}"; echo "[g1] TEST opened by user decision; primary $PRIMARY"
INV_GE="$M3/m32/units_gemma+ent_test.jsonl"; INV_DI="$M3/m32/units_distilled_test.jsonl"; ACU="$G/b18/acu_units_test.jsonl"
# 1. frozen verifiers on the TEST inventories
"$PLAN_PY" "$MCCODE/mc_score_units.py" --units "gemma+ent=$INV_GE" "distilled=$INV_DI" --models "$MC/e1/mc-A-lam1-crossenc-inv_seed*/model" --tag humanfact --scores-out "$G1/scores"
"$PLAN_PY" "$M3CODE/m31_score_pipeline_all.py" --units "gemma+ent=$INV_GE" "distilled=$INV_DI" --ckpts "$G/b21/xenc_seed*" --scores-out "$G1/scores"
# 2. optional promoted variants (set by the G3 decision): union inventory and/or ensemble
if [ "${G1_PROMOTED:-none}" = "union" ]; then
  "$PLAN_PY" "$MCCODE/mc_variants.py" union --units "gemma+ent=$INV_GE" "distilled=$INV_DI" --scores-a "$G1/scores/gemma+ent_humanfact_seed*.jsonl" --scores-b "$G1/scores/distilled_humanfact_seed*.jsonl" --name union --split test --out "$G1/inv"
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "union=$G1/inv/units_union_test.jsonl" --scores-glob "$G1/inv/union_humanfact_seed*.jsonl" --out "$G1/val"
  EXTRA="union_humanfact+div1=$G1/inv/units_union_test.jsonl:$G1/val/union_humanfact_seed*_div1.jsonl"
elif [ "${G1_PROMOTED:-none}" = "ensemble" ]; then
  "$PLAN_PY" "$MCCODE/mc_variants.py" ensemble --units "gemma+ent=$INV_GE" --scores-a "$G1/scores/gemma+ent_humanfact_seed*.jsonl" --scores-b "$G1/scores/gemma+ent_xenc_seed*.jsonl" --tag ens --out "$G1/inv"
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "gemma+ent=$INV_GE" --scores-glob "$G1/inv/gemma+ent_ens_seedens.jsonl" --out "$G1/val"
  EXTRA="ensemble+div1=$INV_GE:$G1/val/gemma+ent_ens_seedens_div1.jsonl"
else EXTRA=""; fi
# 3. the rule in both registered regimes -- consensus (per seed, averaged; the regime of record) and single-model (seed-averaged
#    scores, one file) -- then the one-shot evaluation
div () { "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "$1" --scores-glob "$2" --out "$G1/val"; }
div "gemma+ent=$INV_GE" "$G1/scores/gemma+ent_humanfact_seed*.jsonl"; div "distilled=$INV_DI" "$G1/scores/distilled_humanfact_seed*.jsonl"
div "gemma+ent=$INV_GE" "$G1/scores/gemma+ent_xenc_seed*.jsonl"; div "distilled=$INV_DI" "$G1/scores/distilled_xenc_seed*.jsonl"
for T in humanfact xenc; do for I in gemma+ent distilled; do
  [ "$I" = "gemma+ent" ] && INV="$INV_GE" || INV="$INV_DI"
  "$PLAN_PY" "$MCCODE/mc_variants.py" seedavg --units "$I=$INV" --scores "$G1/scores/${I}_${T}_seed*.jsonl" --tag "$T" --out "$G1/inv"; div "$I=$INV" "$G1/inv/${I}_${T}_seedavg.jsonl"
done; done
SINGLE="humanfact.single+div1=$INV_GE:$G1/val/gemma+ent_humanfact_seedavg_div1.jsonl humanfact.distilled.single+div1=$INV_DI:$G1/val/distilled_humanfact_seedavg_div1.jsonl judgefact.single+div1=$INV_GE:$G1/val/gemma+ent_xenc_seedavg_div1.jsonl judgefact.distilled.single+div1=$INV_DI:$G1/val/distilled_xenc_seedavg_div1.jsonl"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --acu "$ACU" --k 5 10 --n-boot 500 --primary "$PRIMARY" --out "$G1/eval_test" --dump-emitted "$G1/emitted_test.jsonl" --systems \
  "humanfact+div1=$INV_GE:$G1/val/gemma+ent_humanfact_seed*_div1.jsonl" "humanfact.distilled+div1=$INV_DI:$G1/val/distilled_humanfact_seed*_div1.jsonl" \
  "judgefact+div1=$INV_GE:$G1/val/gemma+ent_xenc_seed*_div1.jsonl" "judgefact.distilled+div1=$INV_DI:$G1/val/distilled_xenc_seed*_div1.jsonl" \
  "humanfact=$INV_GE:$G1/scores/gemma+ent_humanfact_seed*.jsonl" "judgefact.distilled=$INV_DI:$G1/scores/distilled_xenc_seed*.jsonl" $SINGLE $EXTRA 2>&1 | grep "\[e2"
(cd "$G1" && find . -type f -name "*.json*" | sort | xargs sha256sum > SHA256SUMS); plan_done "$G1"
