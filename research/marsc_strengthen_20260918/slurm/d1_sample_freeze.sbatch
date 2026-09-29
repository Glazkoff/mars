#!/bin/bash
# MARS-C D1 -- draw and FREEZE the current-emitter blinded precision sample (PREREG Wave D, D1).
#
# Why. The paper's only semantic precision evidence annotates `g3/emitted_validation.jsonl` (2026-09-16 20:11),
# an emission the pipeline no longer produces: after the R07 repair the corrected dump is
# `r07_crossed/emitted_validation_k10.jsonl` (2026-09-17 15:41:22) and the measured top-10 overlap between the
# two is 0.59/0.66/0.63, with only 51/53/48/70 of the 100 annotated items per system surviving. Re-analysing
# the old labels is a selection on rank, not a measurement of the current system.
#
# What this job does. It draws a blinded sample from the CORRECTED dump with the registered sampling instrument
# `mc_human_sample.py` (used UNMODIFIED), and then freezes it: schema contract, id-collision gate against every
# previously issued batch, the document map that makes document-clustered inference possible, and a review
# priority order fixed before any label exists. It reads no label and computes no outcome. Nothing here is a
# scientific result; it is the artifact that a human study can later be run against without post-hoc choice.
#
# Id arithmetic, checked before submission. `mc_human_sample.py` formats ids as f"H{offset+i+1:03d}", which
# SILENTLY produces a 4-digit id past H999 and would break the label app's fixed-width id contract. Issued so
# far: H001..H300 (g6) and H301..H400 (g6b extension), both merged into the live sqlite database at
# https://label.example.org. This batch takes offset 400 and 5 systems x 100 items -> H401..H900. No collision,
# no overflow. d1_freeze.py asserts both against the actual key files rather than trusting this comment.
#SBATCH --job-name=mars-d1-freeze
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source "$PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh"
source "$PLAN_REPO/research/mars3_20260916/slurm/env.sh"
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
OUT="/home/user/results/marsc_strengthen/d1d2/d1"; mkdir -p "$OUT"; plan_receipt "$OUT"

EMITTED="$MC/r07_crossed/emitted_validation_k10.jsonl"
ACU_V="$G/b18/acu_units_validation.jsonl"
PRIOR="$MC/g6/human_sample_key.jsonl $MC/g6/human_sample_key_all.jsonl $MC/g6b/human_sample_key.jsonl"
for f in $EMITTED $ACU_V $PRIOR; do [ -f "$f" ] || { echo "[d1] MISSING $f" >&2; exit 4; }; done

SYSTEMS="humanfact+div1 humanfact.distilled+div1 judgefact+div1 judgefact.distilled+div1 mars2"
PER=100; OFFSET=400; SEED=20260916
STRATA='unknown=0.6,hit=0.2,false_alert=0.2'
ARGV="mc_human_sample.py --emitted $EMITTED --acu $ACU_V --systems $SYSTEMS --per-system $PER --strata $STRATA --seed $SEED --id-offset $OFFSET --out $OUT"

echo "===== D1.1 draw the blinded sample from the CORRECTED emission (registered instrument, unmodified)"
"$PLAN_PY" "$MCCODE/mc_human_sample.py" --emitted "$EMITTED" --acu "$ACU_V" --systems $SYSTEMS \
  --per-system "$PER" --strata "$STRATA" --seed "$SEED" --id-offset "$OFFSET" --out "$OUT"

echo "===== D1.2 freeze it: schema, id-collision gate, document map, review order"
"$PLAN_PY" "$SCODE/d1_freeze.py" --blind "$OUT/human_sample_blind.jsonl" --key "$OUT/human_sample_key.jsonl" \
  --emitted "$EMITTED" --acu "$ACU_V" --prior-keys $PRIOR --sample-argv "$ARGV" \
  --pilot-docs 30 --order-seed 20260918 --out "$OUT"

(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[d1] done -> $OUT"
