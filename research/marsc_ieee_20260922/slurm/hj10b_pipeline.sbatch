#!/bin/bash
# H-J10b -- OmissionBench end to end with the frozen recipe: build the reference rows and decomposition tasks,
# Gemma-4-31B sentence decomposition, spaCy entity units, the deployed human-fact verifier (three seeds, three
# candidate modes), the judge-label twin (b21 xenc, three seeds), the div1 rule, the frozen evaluator with
# strict three-seed manifests, and the observed consultation-clustered contrasts. Registered in
# research/marsc_ieee_20260922/PREREG.md amendment 10 BEFORE submission.
#
#   sbatch research/marsc_ieee_20260922/slurm/hj10b_pipeline.sbatch
#SBATCH --job-name=mars-hj10b
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"; JCODE="$PLAN_REPO/research/marsc_ieee_20260922/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
OMB="${HJ10_DATA:-/home/user/data/omissionbench}"
H="$MCS/hj10b"; mkdir -p "$H/props" "$H/scores" "$H/val" "$H/eval" "$MCS/logs"; plan_receipt "$H"
SEEDS="${MC_SEEDS:-20260906 20260907 20260908}"
for f in "$OMB/pairs/dataset_v2.json" "$OMB/fact_sheets/aci.json" "$OMB/fact_sheets/authored.json"; do [ -f "$f" ] || { echo "[hj10b] MISSING $f"; exit 4; }; done

echo "===== [hj10b] build (census before any scorer)"
"$PLAN_PY" "$JCODE/hj10b_build_omb.py" build --pairs "$OMB/pairs/dataset_v2.json" --sheets "$OMB/fact_sheets" --transcripts "$OMB/transcripts" --out "$H"
ACU="$H/acu_units_omb.jsonl"; TASKS="$H/decompose_tasks_omb.jsonl"; [ -s "$ACU" ] && [ -s "$TASKS" ] || { echo "[hj10b] build wrote nothing"; exit 4; }

echo "===== [hj10b] Gemma-4-31B sentence decomposition (the frozen inventory recipe)"
[ -s "$H/props/props_omb_shard0.jsonl" ] || "$PLAN_PY" "$M3CODE/m32_decompose.py" --tasks "$TASKS" --out "$H/props/props_omb_shard0.jsonl" --shard 0/1 --batch-size 12 2>&1 | grep -E "\[m32" | tail -4
N=$(grep -c . "$H/props/props_omb_shard0.jsonl"); M=$(grep -c . "$TASKS"); [ "$N" -eq "$M" ] || { echo "[hj10b] decomposition wrote $N of $M tasks"; exit 5; }

echo "===== [hj10b] inventory: propositions + spaCy entities"
"$PLAN_PY" "$JCODE/hj10b_build_omb.py" inventory --acu "$ACU" --props "$H/props/props_omb_shard0.jsonl" --out "$H"
INV="$H/units_gemma+ent_omb.jsonl"; [ -s "$INV" ] || { echo "[hj10b] inventory not written"; exit 5; }

echo "===== [hj10b] the deployed human-fact verifier, three seeds, three candidate modes"
for MODE in real shuffled empty; do
  case "$MODE" in real) T="humanfact" ;; shuffled) T="humanfactshuf" ;; empty) T="humanfactempty" ;; esac
  "$PLAN_PY" "$MCCODE/mc_score_units.py" --units "gemma+ent=$INV" --models "$MC/e1/mc-A-lam1-crossenc-inv_seed*/model" \
    --tag "$T" --candidate-mode "$MODE" --shuffle-seed 20260917 --scores-out "$H/scores" 2>&1 | grep -E "\[score" || true
  for S in $SEEDS; do [ -s "$H/scores/gemma+ent_${T}_seed${S}.jsonl" ] || { echo "[hj10b] missing $T seed $S"; exit 5; }; done
done
echo "===== [hj10b] the judge-label twin (b21 xenc), three seeds, real note"
for S in $SEEDS; do
  [ -s "$H/scores/gemma+ent_xenc_seed$S.jsonl" ] || "$PLAN_PY" "$M3CODE/m31_score_pipeline_all.py" --units "gemma+ent=$INV" --ckpts "$G/b21/xenc_seed$S" --scores-out "$H/scores" 2>&1 | grep -E "\[m31|\[score" | tail -2 || true
  [ -s "$H/scores/gemma+ent_xenc_seed$S.jsonl" ] || { echo "[hj10b] missing xenc seed $S"; exit 5; }
done

echo "===== [hj10b] the frozen div1 rule"
"$PLAN_PY" "$MCCODE/mc_diversity.py" --units "gemma+ent=$INV" --cap 1 --scores-glob "$H/scores/gemma+ent_*_seed*.jsonl" --out "$H/val"
roster () { local pre="$1" out="" s f; for s in $SEEDS; do f="$H/val/gemma+ent_${pre}_seed${s}_div1.jsonl"; [ -f "$f" ] || { echo "[hj10b] MISSING $f" >&2; return 1; }; out="${out:+$out,}$f"; done; printf '%s' "$out"; }
SPECS="humanfact+div1=$INV:$(roster humanfact) humanfact.shuffled+div1=$INV:$(roster humanfactshuf) humanfact.empty+div1=$INV:$(roster humanfactempty) judgefact+div1=$INV:$(roster xenc)"
echo "===== [hj10b] evaluation (k=10; strict three-seed manifests; consultation-clustered bootstrap)"
"$PLAN_PY" "$MCCODE/mc_e2_eval.py" --strict-seeds --expect-seeds 3 --acu "$ACU" --k 10 --n-boot 2000 --robust-tests --n-perm 10000 \
  --primary "humanfact+div1" --out "$H/eval/eval_omb" --dump-per-pair "$H/eval/per_pair_omb_k10.jsonl" --dump-emitted "$H/eval/emitted_omb_k10.jsonl" \
  --systems $SPECS 2> >(tee -a "$H/eval/e2_stderr.log" >&2) | grep -E "\[e2"
"$PLAN_PY" "$SCODE/b3_contrast.py" --n-boot 2000 --boot-seed 20260918 --skip-missing \
  --eval "omb_hj10b=$H/eval/eval_omb:$H/eval/per_pair_omb_k10.jsonl" --out "$H/eval/b3_contrasts.json"
(cd "$H" && find . -type f -name "*.json" -not -path "./props/*" | sort | xargs sha256sum > SHA256SUMS); plan_done "$H"
echo "[hj10b] done -> $H/eval"
