#!/bin/bash
# M-G (audit response, 2026-09-28) -- the metric table on common populations, and RoSE MARS-R without gold ACUs.
# Registered in research/mars_metric_20260923/PREREG.md (Amendment M-G) BEFORE submission.
#
#   MG_STAGE=rose   GPU: build the two automatic-unit RoSE arms (mg_rose_autoref.py), decompose the 286 RoSE
#                   references with the frozen M32 Gemma recipe, score MARS-R (MARS-C verifier and FactCG) on both
#                   arms with ma_score.py unmodified, rename the columns (…_autogemma, …_autoa2cu)
#   MG_STAGE=meta   CPU: ma_meta.py four ways over every score file present --
#                   (1) common cohort over the Table 3 columns + pairwise-common contrasts + NI margin 0.05
#                   (2) the same without the three SummEval sources that occur among MARS-C training sources
#                   (3) the default per-column populations without those three sources
#                   (4) the default per-column populations, all documents (reproduces the published table + Holm)
#                   (5) the default populations with every MARS contrast recomputed on the pairs both columns share
#
#   R=$(sbatch --parsable --export=ALL,MG_STAGE=rose <this>)
#   sbatch --dependency=afterany:$R --export=ALL,MG_STAGE=meta --gres=none --cpus-per-task=16 --mem=64G --time=06:00:00 <this>
#SBATCH --job-name=mars-mg
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/results/mars_metric/logs/%x-%j.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"
MM="${MARS_METRIC_OUT:-/home/user/results/mars_metric}"; CODE="$PLAN_REPO/research/mars_metric_20260923/code"
OUT="$MM/mg"; mkdir -p "$OUT/rose" "$OUT/scores" "$OUT/meta"
STAGE="${MG_STAGE:?set MG_STAGE=rose|meta}"

case "$STAGE" in
rose)
  plan_receipt "$OUT/rose"
  "$PLAN_PY" "$CODE/mg_rose_autoref.py" --pairs "$MM/ma/tasks/pairs_ma.jsonl" \
    --a2cu-acus "$MM/ma/scores/scores_a2cu.acus.json" --out "$OUT/rose"
  [ -s "$OUT/rose/props_rose_ref.jsonl" ] || \
    "$PLAN_PY" "$M3CODE/m32_decompose.py" --tasks "$OUT/rose/tasks_rose_ref.jsonl" \
      --out "$OUT/rose/props_rose_ref.jsonl" --shard 0/1 --batch-size 12 2>&1 | grep -E "\[m32" | tail -n 5
  SNAP=$(snap "yaxili96/FactCG-DeBERTa-v3-Large") || true
  for ARM in gemma a2cu; do
    for M in marsR_marsc marsR_factcg; do
      RAW="$OUT/rose/raw_${M}_${ARM}.jsonl"
      B=32; [ "$M" = marsR_marsc ] && B=128
      "$PLAN_PY" "$CODE/ma_score.py" --pairs "$OUT/rose/pairs_rose_${ARM}.jsonl" --metric "$M" --out "$RAW" \
        --snapshot "$SNAP" --ckpts "$MC/e1/mc-A-lam1-crossenc-inv_seed*/model" --props "$OUT/rose/props_rose_ref.jsonl" \
        --batch "$B" 2>&1 | grep -E "^\[ma\]|Error|Traceback" || true
      "$PLAN_PY" - "$RAW" "$M" "auto$ARM" "$OUT/scores/scores_${M}_auto${ARM}.jsonl" <<'PY'
import json, sys
raw, m, tag, out = sys.argv[1:5]
with open(out, "w") as fh:
    for line in open(raw):
        r = json.loads(line); o = {"pair_id": r["pair_id"]}
        for k, v in r.items():
            if k.startswith(m):
                o[m + "_" + tag + k[len(m):]] = v
        fh.write(json.dumps(o) + "\n")
print(f"[mg] {out}")
PY
    done
  done
  plan_done "$OUT/rose" ;;
meta)
  plan_receipt "$OUT/meta"
  "$PLAN_PY" - "$MM/ma/summeval_source/build_manifest.json" "$OUT/meta/summeval_overlap_docs.txt" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))["documents_overlapping_marsc_training_sources"]
open(sys.argv[2], "w").write("\n".join(d) + "\n"); print(f"[mg] {len(d)} overlapping SummEval docs")
PY
  TABLE="marsR_marsc,marsR_factcg,marsR_free,marsP_factcg,rouge1,rouge2,rougeL,bertscore_P,bertscore_R,bertscore_F,bartscore_ref2cand,bartscore_src2cand,alignscore,factcg_whole,a3cu_R,a3cu_F,a2cu_R,qaeval_F1,questeval,questeval_R,unieval_relevance,unieval_consistency,unieval_coherence,unieval_overall,geval_completeness,gevalplus_completeness,geval_faithfulness,gevalplus_faithfulness"
  SC=("$MM/ma/scores/scores_*.jsonl" "$OUT/scores/scores_*.jsonl")
  run () { "$PLAN_PY" "$CODE/ma_meta.py" --pairs "$MM/ma/tasks/pairs_ma.jsonl" --scores "${SC[@]}" \
             --n-boot 2000 --seed 20260918 "$@" 2>&1 | grep -E "^\[ma-meta\] (bar|written)"; }
  run --common-cohort "$TABLE" --pairwise-common --ni-margin 0.05 --out "$OUT/meta/meta_common.json"
  run --common-cohort "$TABLE" --pairwise-common --ni-margin 0.05 --exclude-docs "$OUT/meta/summeval_overlap_docs.txt" \
      --out "$OUT/meta/meta_common_noverlap.json"
  run --ni-margin 0.05 --exclude-docs "$OUT/meta/summeval_overlap_docs.txt" --out "$OUT/meta/meta_default_noverlap.json"
  run --ni-margin 0.05 --out "$OUT/meta/meta_default.json"
  run --pairwise-common --ni-margin 0.05 --out "$OUT/meta/meta_pairwise.json"   # every contrast on the pairs both columns share
  (cd "$OUT" && find meta scores rose -type f -name "*.json*" | sort | xargs sha256sum > meta/SHA256SUMS)
  plan_done "$OUT/meta" ;;
*) echo "[mg] bad stage $STAGE"; exit 2 ;;
esac
echo "[mg] $STAGE done -> $OUT"
