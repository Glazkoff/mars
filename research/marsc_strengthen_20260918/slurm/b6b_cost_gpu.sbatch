#!/bin/bash
# MARS-C B6b -- the GPU half of the stage-level cost account, and the MEASURED amortisation curve.
# Registered: research/marsc_strengthen_20260918/PREREG.md, Wave B / B6. No bar.
#
# What this re-times on the CURRENT node, and why each one is not already in the paper:
#   verification   all THREE seeds of BOTH label sources over BOTH inventories. The G4 table timed ONE
#                  checkpoint per label source, so the published s/pair is one third of the inference cost of
#                  the pipeline of record, which seed-averages three.
#   aggregation    the seed mean. Never priced.
#   off-the-shelf  lex / MiniLM / windowed RoBERTa-MNLI / SummaC-ZS / MiniCheck / AlignScore under the
#                  b21_zeroshot contract. mc_cost.py can only time AutoModelForSequenceClassification at
#                  max_length=512, so the paper compares MARS-C to these verifiers on quality with NO cost
#                  axis -- and SummaC-ZS is the arm that already beats the pipeline of record on recall@10.
#                  AlignScore is expected to fail here (no alignscore package in .venv-mars); a failing
#                  dependency is recorded as a result, it does not abort the job.
#   decomposition  Gemma-4-31B and the distilled LongT5 segmenter as ACTIVE GENERATION. The published
#                  6.637 s/document comes from `sacct -j 4704 -X -o Elapsed` summed over eight array tasks --
#                  allocation wall-clock including model load, tokenizer build and IO, not kernel time.
#
# Slow comparators are unit-capped (--slow-unit-cap) because seconds-per-unit is a rate: it does not depend on
# how many units are timed, and the curve is priced at split-wide unit counts, never at the sample's.
#
# GPU. Validation inventories only; the sealed TEST split is not read.
#SBATCH --job-name=mars-b6b-cost-gpu
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
OUT="$MS/b6b8/b6"; mkdir -p "$OUT"; plan_receipt "$OUT"

INV_GE="$M3/m32/units_gemma+ent_validation.jsonl"
INV_DI="$M3/m32/units_distilled_validation.jsonl"
SEG="$M3/m32/segmenter"
GEMMA=$(snap "google/gemma-4-31B-it")
for f in "$INV_GE" "$INV_DI" "$SEG/config.json"; do
  [ -f "$f" ] || { echo "[b6b] MISSING $f" >&2; exit 4; }
done
for s in 20260906 20260907 20260908; do
  [ -f "$MC/e1/mc-A-lam1-crossenc-inv_seed$s/model/config.json" ] || { echo "[b6b] MISSING humanfact seed $s" >&2; exit 4; }
  [ -f "$G/b21/xenc_seed$s/config.json" ] || { echo "[b6b] MISSING judgefact seed $s" >&2; exit 4; }
done
# trap 7: several hf_cache entries are metadata-only stubs that still resolve to a path and fail at load
if [ -z "$GEMMA" ] || [ ! -f "$GEMMA/config.json" ]; then echo "[b6b] MISSING gemma snapshot" >&2; exit 4; fi
GSZ=$(du -sm --dereference "$GEMMA" 2>/dev/null | awk '{print $1}')
echo "[b6b] gemma snapshot $GEMMA (${GSZ} MiB)"
[ "${GSZ:-0}" -ge 20000 ] || { echo "[b6b] gemma snapshot looks like a metadata-only stub (${GSZ} MiB)" >&2; exit 4; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

set +e
"$PLAN_PY" "$SCODE/b6_cost.py" --stage timings --which gpu \
  --inv-gemma "$INV_GE" --inv-distilled "$INV_DI" \
  --ckpt-human "$MC/e1/mc-A-lam1-crossenc-inv_seed{seed}/model" \
  --ckpt-judge "$G/b21/xenc_seed{seed}" \
  --seeds 20260906,20260907,20260908 \
  --segmenter "$SEG" --gemma-snapshot "$GEMMA" \
  --n-pairs "${B6_N_PAIRS:-200}" --n-docs "${B6_N_DOCS:-50}" --n-docs-gemma "${B6_N_DOCS_GEMMA:-25}" \
  --offshelf "${B6_OFFSHELF:-lex,minilm,nli,summac_zs,minicheck,alignscore}" \
  --offshelf-unit-cap "${B6_UNIT_CAP:-6000}" --slow-unit-cap "${B6_SLOW_CAP:-900}" \
  --timings-name stage_times_gpu.json --out "$OUT"
RC=$?
set -e
echo "[b6b] timings stage exit=$RC (stage results are written incrementally; the curve is built from whatever landed)"
[ -f "$OUT/stage_times_gpu.json" ] || { echo "[b6b] no GPU stage file at all" >&2; exit 5; }

MERGE=""
if [ -f "$OUT/stage_times_cpu.json" ]; then MERGE="$OUT/stage_times_cpu.json"; fi   # set -e safe
"$PLAN_PY" "$SCODE/b6_cost.py" --stage curve \
  --timings-name stage_times_gpu.json --curve-name amortisation_curve.json \
  ${MERGE:+--merge "$MERGE"} --out "$OUT"

(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS)
plan_done "$OUT"
echo "[b6b] done"
