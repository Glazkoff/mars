#!/bin/bash
# MARS-C D2 -- fixed-budget revision study: BUILD the arms, GENERATE the revisions, EXPORT the blinded packets.
#
# Governing proposal Block 3. One revision model, one prompt, six arms that differ only in the feedback facts:
#   none  importance  llmjudge  mars2  marsc  oracle
# Source access, revision calls, candidate count, feedback-token budget and the final word budget are matched
# across arms and asserted programmatically, because the proposal is explicit that MARS-C must not win by
# simply making summaries longer.
#
# This job GENERATES. It computes NO outcome. Every outcome of the revision study (recovered important
# omissions, preservation of previously correct content, unsupported new claims, redundancy, usefulness) needs
# two blinded reviewers plus adjudication. The only machine analogue is the support judge -- the one question
# this project measured to FAIL (kappa 0.4727 on 63,343 cells; it calls 36% of supported facts unsupported) --
# so it is not an admissible substitute and no scoring step exists in this pipeline.
#
# Model. Qwen/Qwen3.5-9B at revision c202236235762e1c871ad0ccb60c8ee5ba337b9a, the revision pinned by the
# acceptance preflight (/home/user/results/marsc_acceptance/preflight_20260918.json, job 5245). The job
# refuses to run against any other revision. Disclosed confound: this is the same model the project uses as its
# forced-choice judge. That is accepted deliberately -- it is the only generation model with a frozen preflight
# pin -- and it costs nothing here because D2's outcomes are human-only by registration.
#
# Trap guarded. Qwen chat templates open a <think> block by default; in an earlier campaign that returned 88%
# empty outputs and a widely-repeated WRONG result. d2_revise.py passes enable_thinking=False AND asserts the
# rendered prompt carries no think opener, refusing to start otherwise.
#SBATCH --job-name=mars-d2-revise
#SBATCH --partition=train
#SBATCH --nodelist=Euler
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source "$PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh"
source "$PLAN_REPO/research/mars3_20260916/slurm/env.sh"
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
OUT="/home/user/results/marsc_strengthen/d1d2/d2"; mkdir -p "$OUT"; plan_receipt "$OUT"

ACU_V="$G/b18/acu_units_validation.jsonl"
EMITTED="$MC/r07_crossed/emitted_validation_k10.jsonl"
INV_GE="$M3/m32/units_gemma+ent_validation.jsonl"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || { echo "[d2] MISSING $f" >&2; return 1; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
IMP=$(roster "$MC/e2imp_all/scores" "gemma+ent_importance_all" ".jsonl") || exit 4
for f in "$ACU_V" "$EMITTED" "$INV_GE"; do [ -f "$f" ] || { echo "[d2] MISSING $f" >&2; exit 4; }; done

PIN_REV="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SNAP="$(snap Qwen/Qwen3.5-9B)"
[ -n "$SNAP" ] || { echo "[d2] no snapshot for Qwen/Qwen3.5-9B" >&2; exit 4; }
SZ=$(du -sh --dereference "$SNAP" | cut -f1)
echo "[d2] model snapshot $SNAP ($SZ)"
case "$SZ" in *G) : ;; *) echo "[d2] snapshot is metadata-only ($SZ) -- refusing" >&2; exit 4;; esac

echo "===== D2.1 build and freeze the six matched arms"
"$PLAN_PY" "$SCODE/d2_build.py" --acu "$ACU_V" --emitted "$EMITTED" --units-gemma "$INV_GE" \
  --importance-scores "$IMP" --k 10 --budget-factor 1.5 --budget-floor 60 \
  --docs-per-resource 50 --summaries-per-doc 2 --min-omitted 2 --pilot-docs 30 --seed 20260918 --out "$OUT"

echo "===== D2.2 generate one greedy revision per (pair, arm)"
"$PLAN_PY" "$SCODE/d2_revise.py" --arms "$OUT/d2_arms.jsonl" --model-snapshot "$SNAP" \
  --expect-revision "$PIN_REV" --out "$OUT/revisions.jsonl" --manifest "$OUT/d2_generation.json" \
  --feedback-token-cap 400 --batch-size 8 --max-input-tokens 6144 --shard 0/1 --limit "${D2_LIMIT:-0}"

echo "===== D2.3 export the blinded review packets (no arm identity, R##### id namespace)"
"$PLAN_PY" "$SCODE/d2_export_packets.py" --arms "$OUT/d2_arms.jsonl" --revisions "$OUT/revisions.jsonl" \
  --review-order "$OUT/d2_review_order.json" --seed 20260918 --out "$OUT"

(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[d2] done -> $OUT"
