#!/bin/bash
# MARS-C T01 -- full ablation grid: inventory x verifier x rule x seed-regime (16 cells) on the sealed TEST split.
# Registered post-hoc SECONDARY endpoint: a re-read of the frozen G1 score files, no new scoring and no tuning.
# Isolates the verifier's contribution at a FIXED inventory and a FIXED emission rule, which is the reviewer-facing
# answer to 'the gain is post-processing'. CPU only.
#SBATCH --job-name=mc-t01-grid
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
G1="$MC/g1_test"; G7="$MC/g7_unisum"
INV_GE="$M3/m32/units_gemma+ent_test.jsonl"; INV_DI="$M3/m32/units_distilled_test.jsonl"; ACU_T="$G/b18/acu_units_test.jsonl"
INV_GE_V="$M3/m32/units_gemma+ent_validation.jsonl"; INV_DI_V="$M3/m32/units_distilled_validation.jsonl"; ACU_V="$G/b18/acu_units_validation.jsonl"
INV_U="$G7/units_gemma+ent_unisum.jsonl"; ACU_U="$G7/acu_units_unisum.jsonl"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[t] REFUSED: MC_TEST_REREAD=yes not set (TEST is sealed; this is a registered post-hoc SECONDARY endpoint on frozen outputs, user decision required)"; exit 3; }
grid_systems () {   # $1 = G1|G7 style root, emits --systems specs for the 2x2x2x2 TEST grid
  echo "\
humanfact+div1=$INV_GE:$G1/val/gemma+ent_humanfact_seed*_div1.jsonl \
humanfact.distilled+div1=$INV_DI:$G1/val/distilled_humanfact_seed*_div1.jsonl \
judgefact+div1=$INV_GE:$G1/val/gemma+ent_xenc_seed*_div1.jsonl \
judgefact.distilled+div1=$INV_DI:$G1/val/distilled_xenc_seed*_div1.jsonl \
humanfact=$INV_GE:$G1/scores/gemma+ent_humanfact_seed*.jsonl \
humanfact.distilled=$INV_DI:$G1/scores/distilled_humanfact_seed*.jsonl \
judgefact=$INV_GE:$G1/scores/gemma+ent_xenc_seed*.jsonl \
judgefact.distilled=$INV_DI:$G1/scores/distilled_xenc_seed*.jsonl \
humanfact.single+div1=$INV_GE:$G1/val/gemma+ent_humanfact_seedavg_div1.jsonl \
humanfact.distilled.single+div1=$INV_DI:$G1/val/distilled_humanfact_seedavg_div1.jsonl \
judgefact.single+div1=$INV_GE:$G1/val/gemma+ent_xenc_seedavg_div1.jsonl \
judgefact.distilled.single+div1=$INV_DI:$G1/val/distilled_xenc_seedavg_div1.jsonl \
humanfact.single=$INV_GE:$G1/inv/gemma+ent_humanfact_seedavg.jsonl \
humanfact.distilled.single=$INV_DI:$G1/inv/distilled_humanfact_seedavg.jsonl \
judgefact.single=$INV_GE:$G1/inv/gemma+ent_xenc_seedavg.jsonl \
judgefact.distilled.single=$INV_DI:$G1/inv/distilled_xenc_seedavg.jsonl"
}

OUT="$MC/t01_grid"; mkdir -p "$OUT"; plan_receipt "$OUT"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --acu "$ACU_T" --k 10 --n-boot 500 --primary "humanfact+div1" \
  --robust-tests --dump-per-pair "$OUT/per_pair_test.jsonl" --out "$OUT/eval_test" --systems $(grid_systems) 2>&1 | grep "\[e2"
sha256sum "$OUT"/eval_test/*.json "$OUT"/per_pair_test.jsonl > "$OUT/SHA256SUMS"; plan_done "$OUT"
