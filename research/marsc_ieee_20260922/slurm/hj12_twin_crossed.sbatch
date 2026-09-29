#!/bin/bash
# H-J12 (audit response, 2026-09-28) -- the label-only twin through every crossed-accuracy analysis that until
# now used only the judge-label PACKAGE (B21, $G/b21/xenc_seed*).
#
# The twin pair is R04: mc-A-lam1-crossenc-r04{human,judge}, the identical E0 train/calib cells, recipe,
# sampler, seeds and step budget; only the coverage label differs (RoSE annotators vs the Gemma-4-31B judge).
# Registered in research/marsc_ieee_20260922/PREREG.md (Amendment H-J12) BEFORE submission.
#
#   (a) crossed accuracy (b2_crossed_families.py, unmodified) on RoSE validation, RoSE TEST and UniSumEval, with
#       the frozen caches of record copied (never written), so every existing family is served from the SAME
#       scores the tables use and only the six twin checkpoint-seeds are scored. Two passes per split:
#       --primary humanfact (the table's margins) and --primary twinhuman (the exact label-only contrast).
#   (b) the lexical-residual analysis (c2_crossed_residual.py, unmodified): surface model fitted on RoSE
#       validation cells per seed, applied unchanged to TEST and to UniSumEval, for the package AND the twins.
#
#   sbatch --export=ALL,MC_TEST_REREAD=yes research/marsc_ieee_20260922/slurm/hj12_twin_crossed.sbatch
#SBATCH --job-name=mars-hj12-twin
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%j.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[hj12] REFUSED: sealed TEST re-read needs MC_TEST_REREAD=yes"; exit 3; }
ROOT="$MCS/hj12/twin"; mkdir -p "$ROOT"; plan_receipt "$ROOT"

HFM="$MC/e1/mc-A-lam1-crossenc-inv_seed*/model"; JFM="$G/b21/xenc_seed*"
THM="$MC/r04_matched/models/mc-A-lam1-crossenc-r04human_seed*/model"
TJM="$MC/r04_matched/models/mc-A-lam1-crossenc-r04judge_seed*/model"
for p in "$THM" "$TJM"; do n=$(ls -d $p 2>/dev/null | wc -l); [ "$n" = 3 ] || { echo "[hj12] expected 3 seeds for $p, got $n"; exit 4; }; done

UNI="$MCS/hj3/docs/docs_unisum.jsonl"
declare -A DOCS=([validation]="$MC/e0/docs_validation.jsonl" [test]="$MC/e0/docs_test.jsonl" [unisum]="$UNI")
declare -A SEED=([validation]="$MCS/b12/validation/cache" [test]="$MCS/b12/test/cache" [unisum]="$MCS/hj3/hcond_unisum/cache")
for S in validation test unisum; do
  [ -f "${DOCS[$S]}" ] || { echo "[hj12] MISSING ${DOCS[$S]}"; exit 4; }
  OUT="$ROOT/crossed_$S"; CACHE="$OUT/cache"; mkdir -p "$CACHE"
  cp -n "${SEED[$S]}/"*.json "$CACHE/"; echo "[hj12] $S seeded cache: $(ls -1 "$CACHE" | wc -l) files"
  for PRIM in humanfact twinhuman; do
    "$PLAN_PY" "$SCODE/b2_crossed_families.py" --docs "${DOCS[$S]}" --cache "$CACHE" \
      --primary "$PRIM" --batch 64 --n-boot 2000 --seed 20260917 --expect-seeds 3 --out "$OUT/primary_$PRIM" \
      --families "humanfact=ckpt:$HFM" "judgefact=ckpt:$JFM" \
        "twinhuman=ckpt:$THM" "twinjudge=ckpt:$TJM" \
        "nli=zs:nli" "summac_zs=zs:summac_zs" \
        "lex_token_recall=lex:token_recall" "lex_rougeL_recall=lex:rougeL_recall" \
        "emb_minilm=emb:sentence-transformers/all-MiniLM-L6-v2" \
        "factcg=zs:factcg" "minicheck7b=zs:minicheck7b" "judge_gemma31=zs:judge_gemma31" \
      2> >(tee -a "$OUT/b2_stderr.log" >&2) | grep -E "^\[b2\]"
  done
done

echo "===== [hj12] lexical residual: RoSE validation -> TEST, and RoSE validation -> UniSumEval"
for PAIR in "test:${DOCS[test]}" "unisum:$UNI"; do
  NAME="${PAIR%%:*}"; DT="${PAIR#*:}"
  for PRIM in humanfact twinhuman; do
    OUT="$ROOT/residual_$NAME/primary_$PRIM"; mkdir -p "$OUT"
    "$PLAN_PY" "$SCODE/c2_crossed_residual.py" \
      --docs-validation "${DOCS[validation]}" --docs-test "$DT" \
      --families "humanfact=$HFM" "judgefact=$JFM" "twinhuman=$THM" "twinjudge=$TJM" \
      --primary "$PRIM" --batch 64 --n-boot 2000 --seed 20260917 --expect-seeds 3 \
      --out "$OUT" 2>&1 | grep -E "^\[c2x"
  done
done

(cd "$ROOT" && find . -type f -name "*.json" ! -path "*/cache/*" | sort | xargs sha256sum > SHA256SUMS)
plan_done "$ROOT"; echo "[hj12] done -> $ROOT"
