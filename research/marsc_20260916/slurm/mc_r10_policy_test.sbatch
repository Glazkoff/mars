#!/bin/bash
# MARS-C R10 -- the importance policy on the held-out test split, as an explicitly POST-HOC SECONDARY endpoint.
#
# Why this exists. E2 registered a learned source-only importance policy and dropped it because, once the
# emission rule was applied, it appeared to cost 0.029 recall@10 on validation. The estimator repair (R09)
# shows that comparison was against a contaminated reference cell: corrected, the policy ADDS
# +0.060 [+0.015,+0.101] on top of the rule on validation. The drop decision was therefore made on a defective
# number and has to be reported as reversed.
#
# Discipline. Nothing is retrained and no threshold is refitted: the three importance taggers are the E2
# checkpoints and tau is the value chosen on the TUNE role before TEST was ever opened. This is still a
# post-hoc read of a split that was scored once, exactly like the registered T01--T08 secondary endpoints, and
# the result is reported as secondary. It does not become the pipeline of record.
#SBATCH --job-name=mc-r10-policy
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[r10] REFUSED: MC_TEST_REREAD=yes not set"; exit 3; }
IMP="$MC/e2imp"; IMP2="$MC/e2imp_all"; G1="$MC/g1_test"
OUT="$MC/r10_policy_test"; mkdir -p "$OUT/scores" "$OUT/val"; plan_receipt "$OUT"
INV_GE="$M3/m32/units_gemma+ent_test.jsonl"; INV_DI="$M3/m32/units_distilled_test.jsonl"; ACU_T="$G/b18/acu_units_test.jsonl"
TAU=$(cat "$IMP/chosen_tau.txt"); echo "[r10] frozen TUNE threshold tau=$TAU"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || { echo "[r10] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }

echo "===== R10.1 the frozen importance taggers on the TEST inventories"
"$PLAN_PY" "$MCCODE/mc_score_importance.py" --ckpts "$IMP2/tagger_seed*" \
  --units "gemma+ent=$INV_GE" "distilled=$INV_DI" --tag importance_all --scores-out "$OUT/scores" 2>&1 | grep -E "\[imp-score|\[m31\]"

echo "===== R10.2 filter every emitter identically at the frozen tau, then the rule"
comb () { "$PLAN_PY" "$MCCODE/mc_importance.py" combine --omission-glob "$1" --importance-glob "$OUT/scores/$2_importance_all_seed*.jsonl" --tau "$TAU" --out "$OUT/val"; }
comb "$G1/scores/gemma+ent_humanfact_seed*.jsonl" "gemma+ent"
comb "$G1/scores/distilled_humanfact_seed*.jsonl" "distilled"
comb "$G1/scores/gemma+ent_xenc_seed*.jsonl"      "gemma+ent"
comb "$G1/scores/distilled_xenc_seed*.jsonl"      "distilled"
div () { "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "$1" --scores-glob "$2" --out "$OUT/val"; }
div "gemma+ent=$INV_GE" "$OUT/val/gemma+ent_humanfact_seed*_imp$TAU.jsonl"
div "distilled=$INV_DI" "$OUT/val/distilled_humanfact_seed*_imp$TAU.jsonl"
div "gemma+ent=$INV_GE" "$OUT/val/gemma+ent_xenc_seed*_imp$TAU.jsonl"
div "distilled=$INV_DI" "$OUT/val/distilled_xenc_seed*_imp$TAU.jsonl"

echo "===== R10.3 the frozen TEST protocol, policy cells beside the pipeline of record"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU_T" --k 5 10 --n-boot 500 \
  --primary "humanfact+div1" --robust-tests --n-perm 10000 --out "$OUT/eval_test" --systems \
  "humanfact+div1=$INV_GE:$(roster "$G1/val" "gemma+ent_humanfact" "_div1.jsonl")" \
  "humanfact.distilled+div1=$INV_DI:$(roster "$G1/val" "distilled_humanfact" "_div1.jsonl")" \
  "judgefact.distilled+div1=$INV_DI:$(roster "$G1/val" "distilled_xenc" "_div1.jsonl")" \
  "judgefact+div1=$INV_GE:$(roster "$G1/val" "gemma+ent_xenc" "_div1.jsonl")" \
  "humanfact+imp+div1=$INV_GE:$(roster "$OUT/val" "gemma+ent_humanfact" "_imp${TAU}_div1.jsonl")" \
  "humanfact.distilled+imp+div1=$INV_DI:$(roster "$OUT/val" "distilled_humanfact" "_imp${TAU}_div1.jsonl")" \
  "judgefact.distilled+imp+div1=$INV_DI:$(roster "$OUT/val" "distilled_xenc" "_imp${TAU}_div1.jsonl")" \
  "judgefact+imp+div1=$INV_GE:$(roster "$OUT/val" "gemma+ent_xenc" "_imp${TAU}_div1.jsonl")" \
  "humanfact+imp=$INV_GE:$(roster "$OUT/val" "gemma+ent_humanfact" "_imp${TAU}.jsonl")" 2>&1 | grep -E "\[e2"
(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[r10] done -> $OUT"
