#!/bin/bash
# MARS-C R06 -- off-the-shelf omission baselines on the sealed TEST split (FULL_REVIEW_2026-09-17 item 6:
# "direct omission baselines / a standard NLI verifier are missing").
#
# Every baseline is an EMITTER over the SAME frozen inventory and goes through the SAME emission rule and the
# SAME evaluator as MARS-C, so the only thing that varies is the verifier: a windowed RoBERTa-large-MNLI
# entailment check (the standard NLI verifier), SummaC-ZS over DeBERTa-large-MNLI, MiniCheck-Flan-T5-Large,
# AlignScore-large, a sentence-embedding max-similarity check, and two lexical-recall counters that bound what
# a model-free reader gets. Omission score = 1 - support, zero training, no labels of any kind.
#SBATCH --job-name=mc-r06-offshelf
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[r06] REFUSED: MC_TEST_REREAD=yes not set"; exit 3; }
OUT="$MC/r06_offshelf"; mkdir -p "$OUT/scores" "$OUT/val"; plan_receipt "$OUT"
INV_GE="$M3/m32/units_gemma+ent_test.jsonl"; INV_DI="$M3/m32/units_distilled_test.jsonl"; ACU_T="$G/b18/acu_units_test.jsonl"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
SYS="${MC_R06_SYS:-lex,minilm,nli,summac_zs,minicheck,alignscore}"

echo "===== R06.1 score both TEST inventories with every off-the-shelf verifier"
"$PLAN_PY" "$CODE/b21_zeroshot.py" --unit-set m2 --units "$INV_DI" --units-name distilled --systems "$SYS" --out "$OUT/scores" 2>&1 | grep -E "\[b21-zs"
"$PLAN_PY" "$CODE/b21_zeroshot.py" --unit-set m2 --units "$INV_GE" --units-name "gemma+ent" --systems "$SYS" --out "$OUT/scores" 2>&1 | grep -E "\[b21-zs"

echo "===== R06.2 the identical emission rule"
div () { "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "$1" --scores-glob "$2" --out "$OUT/val"; }
div "gemma+ent=$INV_GE" "$OUT/scores/gemma+ent_*.jsonl"
div "distilled=$INV_DI" "$OUT/scores/distilled_*.jsonl"

echo "===== R06.3 the frozen TEST protocol, MARS-C beside the baselines"
G1="$MC/g1_test"
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || { echo "[r06] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
SPECS="humanfact+div1=$INV_GE:$(roster "$G1/val" "gemma+ent_humanfact" "_div1.jsonl") \
judgefact.distilled+div1=$INV_DI:$(roster "$G1/val" "distilled_xenc" "_div1.jsonl") \
humanfact.distilled+div1=$INV_DI:$(roster "$G1/val" "distilled_humanfact" "_div1.jsonl") \
humanfact.single+div1=$INV_GE:$G1/val/gemma+ent_humanfact_seedavg_div1.jsonl"
for f in "$OUT"/val/gemma+ent_*_div1.jsonl; do
  b=$(basename "$f" _div1.jsonl); n=${b#gemma+ent_}
  SPECS="$SPECS zs.$n+div1=$INV_GE:$f"
done
for f in "$OUT"/val/distilled_*_div1.jsonl; do
  b=$(basename "$f" _div1.jsonl); n=${b#distilled_}
  SPECS="$SPECS zs.distilled.$n+div1=$INV_DI:$f"
done
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU_T" --k 5 10 --n-boot 500 \
  --primary "humanfact+div1" --robust-tests --n-perm 10000 --out "$OUT/eval_test" --systems $SPECS 2>&1 | grep -E "\[e2"
(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[r06] done -> $OUT"
