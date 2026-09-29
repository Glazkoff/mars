#!/bin/bash
# MARS-C strengthening B4d -- the div1 emission rule, the held-out evaluation, and the registered bar.
#
# The emission rule is applied UNCHANGED (mc_diversity.py --cap 1: one unit per source sentence, every later
# unit pushed by exactly -5). The threshold learned on the development extractors is not refitted and no
# selection of any kind is made on the unseen extractor's output.
#
# One evaluation call carries all fourteen arms, so the development and the unseen extractor are measured on
# the SAME pair set with the SAME document-clustered bootstrap: six unseen-extractor arms (human-fact and
# judge-label verifiers x real / shuffled / no summary) and eight frozen development arms (the same six on the
# gemma+ent inventory, plus the two distilled cells). Every score source is an EXPLICIT three-seed comma
# manifest with --strict-seeds --expect-seeds 3; no glob reaches mc_e2_eval.py. --k is 10 alone so that
# --dump-emitted fires at k=10.
#
# THE BAR, declared before execution (PREREG Wave B / B4): cross-extractor transfer is evidence for
# deployment-distribution learning only if the MARS-C ordering on the DEVELOPMENT extractors reproduces on the
# unseen one with a document-clustered interval excluding zero. Two orderings are named, both before the run:
#   O1  supervision ordering: human-fact verifier > judge-label verifier, on recall@10 and omission precision.
#   O2  candidate-conditioning ordering: real summary > shuffled and > no summary, on omission precision
#       (the endpoint that charges for false alarms; recall@k is expected to move the other way and is
#       reported beside it).
# and the difference-in-differences between the unseen and the development extractor is reported for each --
# there, a interval that INCLUDES zero is what reproduction looks like. If an ordering does not reproduce, the
# transfer claim is withdrawn, not rescued by a post-hoc adjustment.
#SBATCH --job-name=mars-b4d-eval
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A.out
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[b4d] REFUSED: MC_TEST_REREAD=yes not set (sealed TEST split)"; exit 3; }
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
OUT="${B4_OUT:-/home/user/results/marsc_strengthen/b4}"; mkdir -p "$OUT/val"
G1="$MC/g1_test"; R3="$MC/r03_sumblind"
INV_Q="$OUT/inv/units_qwenprops+ent_test.jsonl"; INV_GE="$M3/m32/units_gemma+ent_test.jsonl"
INV_DI="$M3/m32/units_distilled_test.jsonl"; ACU="$G/b18/acu_units_test.jsonl"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
for f in "$INV_Q" "$INV_GE" "$INV_DI" "$ACU"; do [ -s "$f" ] || { echo "[b4d] MISSING $f"; exit 4; }; done

echo "===== B4d.1 the emission rule, applied unchanged to the unseen extractor's scores"
for TAG in humanfact humanfactshuffled humanfactempty xenc xencshuffled xencempty; do
  n=$(find "$OUT/scores" -maxdepth 1 -name "qwenprops+ent_${TAG}_seed*.jsonl" | wc -l | tr -d " ")
  [ "$n" = "3" ] || { echo "[b4d] expected 3 score files for tag $TAG, found $n"; exit 4; }
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "qwenprops+ent=$INV_Q" \
    --scores-glob "$OUT/scores/qwenprops+ent_${TAG}_seed*.jsonl" --out "$OUT/val" --cap 1
done

roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}";
  [ -f "$f" ] || { echo "[b4d] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
need () { local s f; for s in $SEEDS; do f="$1/${2}_seed${s}${3}"; [ -f "$f" ] || { echo "[b4d] MISSING $f"; exit 4; }; done; }
for spec in "$OUT/val qwenprops+ent_humanfact" "$OUT/val qwenprops+ent_humanfactshuffled" "$OUT/val qwenprops+ent_humanfactempty" \
            "$OUT/val qwenprops+ent_xenc" "$OUT/val qwenprops+ent_xencshuffled" "$OUT/val qwenprops+ent_xencempty" \
            "$G1/val gemma+ent_humanfact" "$G1/val gemma+ent_xenc" "$G1/val distilled_humanfact" "$G1/val distilled_xenc" \
            "$R3/val gemma+ent_humanfactshuffled" "$R3/val gemma+ent_humanfactempty" \
            "$R3/val gemma+ent_xencshuffled" "$R3/val gemma+ent_xencempty"; do
  set -- $spec; need "$1" "$2" "_div1.jsonl"
done
echo "[b4d] all 14 x 3 = 42 emitter score files present"

echo "===== B4d.2 held-out evaluation, k=10, 2000 document-clustered bootstrap resamples"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU" --k 10 --n-boot 2000 \
  --robust-tests --primary "unseen.qwen.humanfact+div1" --out "$OUT/eval_test" \
  --dump-emitted "$OUT/emitted_test_k10.jsonl" --dump-per-pair "$OUT/per_pair_test_k10.jsonl" --systems \
  "unseen.qwen.humanfact+div1=$INV_Q:$(roster "$OUT/val" "qwenprops+ent_humanfact" "_div1.jsonl")" \
  "unseen.qwen.judgefact+div1=$INV_Q:$(roster "$OUT/val" "qwenprops+ent_xenc" "_div1.jsonl")" \
  "unseen.qwen.humanfact.shuffled+div1=$INV_Q:$(roster "$OUT/val" "qwenprops+ent_humanfactshuffled" "_div1.jsonl")" \
  "unseen.qwen.humanfact.empty+div1=$INV_Q:$(roster "$OUT/val" "qwenprops+ent_humanfactempty" "_div1.jsonl")" \
  "unseen.qwen.judgefact.shuffled+div1=$INV_Q:$(roster "$OUT/val" "qwenprops+ent_xencshuffled" "_div1.jsonl")" \
  "unseen.qwen.judgefact.empty+div1=$INV_Q:$(roster "$OUT/val" "qwenprops+ent_xencempty" "_div1.jsonl")" \
  "dev.gemma.humanfact+div1=$INV_GE:$(roster "$G1/val" "gemma+ent_humanfact" "_div1.jsonl")" \
  "dev.gemma.judgefact+div1=$INV_GE:$(roster "$G1/val" "gemma+ent_xenc" "_div1.jsonl")" \
  "dev.gemma.humanfact.shuffled+div1=$INV_GE:$(roster "$R3/val" "gemma+ent_humanfactshuffled" "_div1.jsonl")" \
  "dev.gemma.humanfact.empty+div1=$INV_GE:$(roster "$R3/val" "gemma+ent_humanfactempty" "_div1.jsonl")" \
  "dev.gemma.judgefact.shuffled+div1=$INV_GE:$(roster "$R3/val" "gemma+ent_xencshuffled" "_div1.jsonl")" \
  "dev.gemma.judgefact.empty+div1=$INV_GE:$(roster "$R3/val" "gemma+ent_xencempty" "_div1.jsonl")" \
  "dev.distilled.humanfact+div1=$INV_DI:$(roster "$G1/val" "distilled_humanfact" "_div1.jsonl")" \
  "dev.distilled.judgefact+div1=$INV_DI:$(roster "$G1/val" "distilled_xenc" "_div1.jsonl")" 2>&1 | grep -E "^\[e2"

echo "===== B4d.3 the registered orderings (declared direction: the first system is expected to be HIGHER)"
"$PLAN_PY" "$SCODE/b4_ordering.py" --per-pair "$OUT/per_pair_test_k10.jsonl" --k 10 \
  --n-boot 2000 --seed 20260918 --direction positive --out "$OUT/ordering_declared.json" --contrast \
  "O1.supervision.unseen|unseen.qwen.humanfact+div1|unseen.qwen.judgefact+div1" \
  "O1.supervision.dev.gemma|dev.gemma.humanfact+div1|dev.gemma.judgefact+div1" \
  "O1.supervision.dev.distilled|dev.distilled.humanfact+div1|dev.distilled.judgefact+div1" \
  "O2.conditioning.unseen.shuffled|unseen.qwen.humanfact+div1|unseen.qwen.humanfact.shuffled+div1" \
  "O2.conditioning.unseen.empty|unseen.qwen.humanfact+div1|unseen.qwen.humanfact.empty+div1" \
  "O2.conditioning.dev.shuffled|dev.gemma.humanfact+div1|dev.gemma.humanfact.shuffled+div1" \
  "O2.conditioning.dev.empty|dev.gemma.humanfact+div1|dev.gemma.humanfact.empty+div1" \
  "O2.conditioning.unseen.judge.shuffled|unseen.qwen.judgefact+div1|unseen.qwen.judgefact.shuffled+div1" \
  "O2.conditioning.dev.judge.shuffled|dev.gemma.judgefact+div1|dev.gemma.judgefact.shuffled+div1"

echo "===== B4d.4 attenuation: does the ordering CHANGE between the development and the unseen extractor?"
echo "        (here an interval that INCLUDES zero is what ordering reproduction looks like)"
"$PLAN_PY" "$SCODE/b4_ordering.py" --per-pair "$OUT/per_pair_test_k10.jsonl" --k 10 \
  --n-boot 2000 --seed 20260918 --direction either --out "$OUT/ordering_attenuation.json" \
  --contrast \
  "L.transfer.humanfact|unseen.qwen.humanfact+div1|dev.gemma.humanfact+div1" \
  "L.transfer.judgefact|unseen.qwen.judgefact+div1|dev.gemma.judgefact+div1" \
  --did \
  "DID.supervision|unseen.qwen.humanfact+div1|unseen.qwen.judgefact+div1|dev.gemma.humanfact+div1|dev.gemma.judgefact+div1" \
  "DID.conditioning.shuffled|unseen.qwen.humanfact+div1|unseen.qwen.humanfact.shuffled+div1|dev.gemma.humanfact+div1|dev.gemma.humanfact.shuffled+div1" \
  "DID.conditioning.empty|unseen.qwen.humanfact+div1|unseen.qwen.humanfact.empty+div1|dev.gemma.humanfact+div1|dev.gemma.humanfact.empty+div1"

(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[b4d] done"
