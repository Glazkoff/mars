#!/bin/bash
# MARS-C G4 -- cost account: verifier seconds per pair over each inventory (200 validation pairs) + inventory build cost from receipts.
#SBATCH --job-name=mc-g4-cost
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/home/user/results/marsc/logs/%x-%A.out
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"; G4="$MC/g4"; mkdir -p "$G4"; plan_receipt "$G4"
DEC=$(sacct -j 4704 -X -o Elapsed --noheader | awk -F: '{s+=$1*3600+$2*60+$3} END {print s}')        # Gemma decomposition, 8 shards, 1986 documents
DIS=$(sacct -j 4706 -X -o Elapsed --noheader | awk -F: '{s+=$1*3600+$2*60+$3} END {print s}')        # distilled segmenter: training + prediction over the same documents
echo "[g4] decomposition GPU seconds $DEC over 1986 documents; distill job seconds $DIS"
"$PLAN_PY" "$MCCODE/mc_cost.py" --units "gemma+ent=$M3/m32/units_gemma+ent_validation.jsonl" "distilled=$M3/m32/units_distilled_validation.jsonl" "spacy=$M3/m32/units_spacy_validation.jsonl" \
  --verifiers "humanfact=$MC/e1/mc-A-lam1-crossenc-inv_seed20260906/model" "judgefact=$G/b21/xenc_seed20260906" --n-pairs 200 \
  --inventory-cost "gemma+ent=$DEC:1986" "gemma=$DEC:1986" "distilled=$DIS:1986" --out "$G4"
plan_done "$G4"
