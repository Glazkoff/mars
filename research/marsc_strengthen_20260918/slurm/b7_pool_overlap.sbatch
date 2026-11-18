#!/bin/bash
# MARS-C B7 -- QAPyramid's REAL overlap with our RoSE pool, and resolution of two unverified pool claims.
# Registered: research/marsc_strengthen_20260918/PREREG.md, Wave B / B7.
#
# The G5 census line "QAPyramid: 0 text records, 0 overlap with our sources" is a MEASUREMENT ARTEFACT:
# mc_pool_feasibility.py's find_text() wants a long string under a text-like top-level key, and QAPyramid's
# raw_data/*.json are dict-of-dict keyed by DOCUMENT ID, so the overlap was never actually computed. QAPyramid
# is CNN/DM-based and rose_cnndm is in the MARS pool, so the overlap is plausibly large. This job measures it.
#
# mc_pool_feasibility.py is NOT modified; the corrected extraction lives in research/marsc_strengthen_20260918/code.
#
# Needs the network for the hub/arXiv probes, so HF_HUB_OFFLINE and friends are unset here; the decisive
# document-id join is local and runs regardless of network state.
#
# SEALED-TEST GATE: the per-role contamination table reads the TEST-role document sources (ids and text only,
# no labels, no scores, no model selection). PREREG.md is the dated amendment that authorises this.
#SBATCH --job-name=mars-b7-pooloverlap
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# measured: 00:00:05 for the id join + QAPyramid hub fetch (jobs 5250, 5266); 00:20:00 is that plus
# generous headroom for the extra pool fetches. A long --time on this box blocks CPU backfill.
#SBATCH --time=00:20:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[b7] REFUSED"; exit 3; }
OUT="$MS/b5b7/b7"; mkdir -p "$OUT"; plan_receipt "$OUT"

for f in "$MC/e0/docs_train.jsonl" "$MC/e0/docs_validation.jsonl" "$MC/e0/docs_test.jsonl" \
         "$MC/e0/docs_calib.jsonl" "$MC/e0/docs_tune.jsonl"; do
  [ -f "$f" ] || { echo "[b7] MISSING $f" >&2; exit 4; }
done
QAP_CACHED="$MC/g5/QAPyramid"
[ -d "$QAP_CACHED" ] && echo "[b7] using the read-only G5 clone at $QAP_CACHED" || QAP_CACHED=""

# the compute node has internet; the offline defaults are only a defensive baseline from common.sh
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE

"$PLAN_PY" "$SCODE/b7_pool_overlap.py" \
  --our-sources "train=$MC/e0/docs_train.jsonl" "validation=$MC/e0/docs_validation.jsonl" \
                "test=$MC/e0/docs_test.jsonl" "calib=$MC/e0/docs_calib.jsonl" "tune=$MC/e0/docs_tune.jsonl" \
  --qapyramid-cached "$QAP_CACHED" \
  --hf-qapyramid "shiyue/QAPyramid" \
  --hf-probe "ComposoAI/OmissionBench" \
  --arxiv-probe "2608.31016" \
  --jaccard-floor 0.5 --max-hf-files 12 \
  --out "$OUT"

(cd "$OUT" && find . -type f -name "*.json" -not -path "./hf/*" -not -path "./QAPyramid/*" | sort | xargs sha256sum > SHA256SUMS); plan_done "$OUT"
echo "[b7] done"
