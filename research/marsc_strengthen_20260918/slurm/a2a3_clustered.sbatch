#!/bin/bash
# MARS-C strengthening campaign -- Wave A blocks A2 / A3 / A3b / A8 (research/marsc_strengthen_20260918/PREREG.md).
#
# Four repairs of the project's confidence-interval machinery. None carries a bar.
#   A2  the R07 crossed inventory x verifier bootstrap resamples pair_id in a variable NAMED by_doc; RoSE puts
#       8-12 summaries of one source document in as independent units. Re-estimated on common.doc_key(source).
#       Declared before execution: the interaction sits 0.0019 from zero and may cross it. Whatever it does is
#       what gets reported.
#   A3  the human precision study resamples annotated items inside status strata. Re-estimated on documents,
#       with the distinct-cluster count beside every interval and an explicit degeneracy flag. Declared before
#       execution: at rank <= 10 the agreement-conditioned cells are ~47/40/44/53 items over ~37/36/37/42
#       documents, so the intervals will get materially wider and a percentile bootstrap may be degenerate.
#       Run over all three published rank-recovery configurations (as-annotated, corrected emitter, reweighted).
#   A3b the annotator disclosure: each annotator's Q1 and Q2 answer distributions, the 2x2 contingency tables
#       and Cohen's kappa for Q1, Q2 and the composite verdict, with raw counts so a reader can recompute.
#   A8  the confidence-interval site audit: every interval site in the MARS-C / gate code and scripts/plan2026,
#       classified by what it actually resamples, with any unmatched scan hit reported as UNCLASSIFIED.
#
# CPU only. Reads validation-pool artefacts and source code; touches no sealed TEST split, so no MC_TEST_REREAD
# gate is required. Writes only under $MC_S/a2a3.
#SBATCH --job-name=mars-a2a3-clustered
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MC_S="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
OUT="$MC_S/a2a3"; mkdir -p "$OUT"; plan_receipt "$OUT"

R7="$MC/r07_crossed"; G6="$MC/g6"; R2="$MC/r02_human_rank"; G3="$MC/g3"
ACU_V="$G/b18/acu_units_validation.jsonl"          # pair_id + source, the document-cluster key for A3
A2A3 () { "$PLAN_PY" "$SCODE/a2a3_clustered.py" "$@"; }

need () { for f in "$@"; do [ -e "$f" ] || { echo "[mcs-a2a3] MISSING $f" >&2; exit 4; }; done; }
need "$R7/tasks.jsonl" "$R7/analysis/crossed_precision.json" "$ACU_V" \
     "$G6/human_sample_key_all.jsonl" "$G6/labels_export_all.jsonl" \
     "$G3/emitted_validation.jsonl" "$G3/eval_validation/e2_emitted.json" \
     "$R7/emitted_validation_k10.jsonl" "$R7/eval_validation/e2_emitted.json"
ls "$R7"/judgments/gemma_coverage_shard*.jsonl >/dev/null || { echo "[mcs-a2a3] MISSING r07 judgments" >&2; exit 4; }

echo "===== A3b  annotator disclosure (run FIRST: it decides what the paper may say about 'two annotators')"
A2A3 a3b --key "$G6/human_sample_key_all.jsonl" --export "$G6/labels_export_all.jsonl" --out "$OUT/a3b"

echo "===== A8  confidence-interval site audit"
A2A3 a8 --out "$OUT/a8"

echo "===== A2  R07 crossed 2x2, document-clustered bootstrap (cells and judgments exactly as mc_r07c ran them)"
A2A3 a2 --tasks "$R7/tasks.jsonl" \
  --judgments "$R7/judgments/gemma_coverage_shard*.jsonl" --max-rank 10 --n-boot 5000 --seed 20260917 \
  --cells "gemma+ent:human=humanfact+div1" "gemma+ent:judge=judgefact+div1" \
          "distilled:human=humanfact.distilled+div1" "distilled:judge=judgefact.distilled+div1" \
  --extra-systems mars2 --out "$OUT/a2"

echo "===== A3.1 human precision, document-clustered -- as-annotated emitter and as-annotated status weights"
A2A3 a3 --emitted "$G3/emitted_validation.jsonl" \
  --key "$G6/human_sample_key_all.jsonl" --export "$G6/labels_export_all.jsonl" \
  --e2-json "$G3/eval_validation/e2_emitted.json" --pair-source "$ACU_V" \
  --budgets 10 20 --n-boot 10000 --seed 20260917 --out "$OUT/a3_as_annotated"

echo "===== A3.2 human precision, document-clustered -- CORRECTED emitter and CORRECTED status weights (R02b.2)"
A2A3 a3 --emitted "$R7/emitted_validation_k10.jsonl" \
  --key "$G6/human_sample_key_all.jsonl" --export "$G6/labels_export_all.jsonl" \
  --e2-json "$R7/eval_validation/e2_emitted.json" --pair-source "$ACU_V" \
  --budgets 10 --n-boot 10000 --seed 20260917 --out "$OUT/a3_corrected"

echo "===== A3.3 human precision, document-clustered -- as-annotated emitter, CORRECTED status weights (R02b.3)"
A2A3 a3 --emitted "$G3/emitted_validation.jsonl" \
  --key "$G6/human_sample_key_all.jsonl" --export "$G6/labels_export_all.jsonl" \
  --e2-json "$R7/eval_validation/e2_emitted.json" --pair-source "$ACU_V" \
  --budgets 10 --n-boot 10000 --seed 20260917 --out "$OUT/a3_reweighted"   # r07's e2 json carries k=10 only

echo "===== provenance: the pair-level numbers these repairs supersede"
"$PLAN_PY" - "$R7/analysis/crossed_precision.json" "$R2/human_precision_ranked.json" \
  "$R2/corrected/human_precision_ranked.json" "$R2/reweighted/human_precision_ranked.json" \
  "$OUT/superseded_inputs.json" <<'PY'
import hashlib, json, sys
paths = sys.argv[1:-1]
out = {"note": "sha256 of the pair-level / item-level outputs that A2 and A3 supersede", "inputs": {}}
for p in paths:
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    out["inputs"][p] = h
    print(f"[mcs-a2a3] superseded {p} sha256 {h}", flush=True)
open(sys.argv[-1], "w").write(json.dumps(out, indent=1))
PY

(cd "$OUT" && find . -type f -name "*.json" | sort | xargs sha256sum > SHA256SUMS)
plan_done "$OUT"
echo "[mcs-a2a3] done"
