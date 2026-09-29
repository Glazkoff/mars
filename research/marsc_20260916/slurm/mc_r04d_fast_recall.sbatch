#!/bin/bash
# MARS-C R04d -- the matched label-source control on the EMITTED endpoint, Gemma+ent only (the pipeline of
# record's inventory), so the decisive recall/precision comparison does not wait on the full 24-pass scoring
# of R04c. Same trained twins, same frozen protocol.
#SBATCH --job-name=mc-r04d-fast
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[r04d] REFUSED"; exit 3; }
OUT="$MC/r04_matched"; mkdir -p "$OUT/scores_d" "$OUT/val_d"
INV_GE="$M3/m32/units_gemma+ent_test.jsonl"; ACU_T="$G/b18/acu_units_test.jsonl"; G1="$MC/g1_test"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
for LAB in human judge; do
  TAG="mc-A-lam1-crossenc-r04${LAB}"
  "$PLAN_PY" "$MCCODE/mc_score_units.py" --units "gemma+ent=$INV_GE" --models "$OUT/models/${TAG}_seed*/model" \
    --tag "r04d${LAB}" --scores-out "$OUT/scores_d" 2>&1 | grep -E "\[score"
  "$PLAN_PY" "$MCCODE/mc_diversity.py" --units "gemma+ent=$INV_GE" --scores-glob "$OUT/scores_d/gemma+ent_r04d${LAB}_seed*.jsonl" --out "$OUT/val_d"
done
roster () { local dir="$1" pre="$2" suf="$3" out="" s f; for s in $SEEDS; do f="$dir/${pre}_seed${s}${suf}"; [ -f "$f" ] || { echo "[r04d] MISSING $f" >&2; exit 4; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU_T" --k 10 --n-boot 2000 \
  --primary "matched.human+div1" --robust-tests --n-perm 10000 --out "$OUT/eval_test_d" --systems \
  "matched.human+div1=$INV_GE:$(roster "$OUT/val_d" "gemma+ent_r04dhuman" "_div1.jsonl")" \
  "matched.judge+div1=$INV_GE:$(roster "$OUT/val_d" "gemma+ent_r04djudge" "_div1.jsonl")" \
  "deployed.humanfact+div1=$INV_GE:$(roster "$G1/val" "gemma+ent_humanfact" "_div1.jsonl")" \
  "deployed.judgefact+div1=$INV_GE:$(roster "$G1/val" "gemma+ent_xenc" "_div1.jsonl")" 2>&1 | grep -E "^\[e2 k"
echo "[r04d] done"
