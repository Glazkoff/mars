#!/bin/bash
# W6 -- Shannon Score and BLANC-help on SummEval under the counter suite (the recovery family audits itself).
# Registered: MARS_WIN_HYPOTHESES_2026-09-09.md Sec. W6; pulled forward by MARS_PUBLICATION_PLAN_2026-09-20.md
# Sec. 3.2 (Paper A). No sealed split is touched: SummEval is a public human-annotation benchmark and this job
# reads no MARS-C inventory, so no MC_TEST_REREAD gate applies.
#
# Pilot (10 documents):
#   sbatch --export=ALL,W6_LIMIT_DOCS=10 research/marsc_strengthen_20260918/slurm/w6_recovery_audit.sbatch
# Full:
#   sbatch research/marsc_strengthen_20260918/slurm/w6_recovery_audit.sbatch
#
#SBATCH --job-name=mars-w6-recovery
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh

SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
MCS="${MARSC_STRENGTHEN_OUT:-/home/user/results/marsc_strengthen}"
OUT="$MCS/w6"; mkdir -p "$OUT" "$MCS/logs"

# SummEval (mteb/summeval) and bert-base-uncased are fetched on first use and then cached; this node has
# outbound access, unlike the cluster-A cluster the slurm/ README was written for.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE || true

GPT2=$(snap "openai-community/gpt2") || true
if [ -z "$GPT2" ] || [ ! -f "$GPT2/config.json" ]; then echo "[w6] MISSING gpt2 snapshot"; exit 4; fi

plan_receipt "$OUT"
echo "[w6] gpt2=$GPT2 out=$OUT"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

LIMIT_ARG=""
[ -n "${W6_LIMIT_DOCS:-}" ] && LIMIT_ARG="--limit-docs $W6_LIMIT_DOCS"

"$PLAN_PY" "$SCODE/w6_recovery_family_audit.py" \
  --out "$OUT" --gpt2-snapshot "$GPT2" \
  --doc-tokens "${W6_DOC_TOKENS:-500}" --batch "${W6_BATCH:-8}" \
  --blanc-batch "${W6_BLANC_BATCH:-128}" \
  --n-boot "${W6_NBOOT:-2000}" $LIMIT_ARG

plan_done "$OUT"
echo "[w6] done -> $OUT"
