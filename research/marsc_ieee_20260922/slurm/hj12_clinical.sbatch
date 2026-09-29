#!/bin/bash
# H-J12 (audit response, 2026-09-28) -- random and lead-position top-10 baselines on the clinical inventory (F15).
# Registered in research/marsc_ieee_20260922/PREREG.md (Amendment H-J12, part e) BEFORE submission. CPU only: the
# verifier's per-unit scores of record (hj10b) are reused, the baselines are written over the same inventory, and
# every emitter goes through the frozen div1 rule (--cap 1, as hj10b) and one evaluator call.
#SBATCH --job-name=mars-hj12-clin
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%j.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
MCCODE="$PLAN_REPO/research/marsc_20260916/code"; SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
JCODE="$PLAN_REPO/research/marsc_ieee_20260922/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
H="$MCS/hj10b"; OUT="$MCS/hj12/clinical"; mkdir -p "$OUT/scores" "$OUT/val" "$OUT/eval"; plan_receipt "$OUT"
INV="$H/units_gemma+ent_omb.jsonl"; ACU="$H/acu_units_omb.jsonl"
SEEDS="20260906 20260907 20260908"
"$PLAN_PY" "$JCODE/hj12_clinical_baselines.py" --units "gemma+ent=$INV" --out "$OUT/scores"
"$PLAN_PY" "$MCCODE/mc_diversity.py" --units "gemma+ent=$INV" --cap 1 --scores-glob "$OUT/scores/gemma+ent_*.jsonl" --out "$OUT/val"
roster () { local dir="$1" pre="$2" out="" s f; shift 2; for s in "$@"; do f="$dir/gemma+ent_${pre}_seed${s}_div1.jsonl"; [ -f "$f" ] || { echo "[hj12c] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
SPECS="humanfact+div1=$INV:$(roster "$H/val" humanfact $SEEDS)"
SPECS="$SPECS humanfact.shuffled+div1=$INV:$(roster "$H/val" humanfactshuf $SEEDS)"
SPECS="$SPECS humanfact.empty+div1=$INV:$(roster "$H/val" humanfactempty $SEEDS)"
SPECS="$SPECS random+div1=$INV:$(roster "$OUT/val" random 1 2 3)"
SPECS="$SPECS leadpos+div1=$INV:$OUT/val/gemma+ent_leadpos_seed0_div1.jsonl"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU" --k 10 --n-boot 2000 \
  --primary "humanfact+div1" --out "$OUT/eval/eval_omb" --dump-per-pair "$OUT/eval/per_pair_omb_k10.jsonl" \
  --systems $SPECS 2>&1 | grep -E "\[e2"
"$PLAN_PY" "$SCODE/b3_contrast.py" --n-boot 2000 --boot-seed 20260918 --skip-missing \
  --eval "omb_hj12=$OUT/eval/eval_omb:$OUT/eval/per_pair_omb_k10.jsonl" --out "$OUT/eval/b3_contrasts.json"
(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[hj12c] done -> $OUT/eval"
