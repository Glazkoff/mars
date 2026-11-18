#!/bin/bash
# M-F smoke -- the mars package with the real components on 40 meta-evaluation pairs: FactCG-contract verifier for
# P, the three deployed MARS-C seeds for R, the distilled LongT5 segmenter as the inventory, controls on. Validates
# the model-loading paths only; no number from it enters any bar.
#SBATCH --job-name=mars-mf-smoke
#SBATCH --partition=infer
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/home/user/results/mars_metric/logs/%x-%A.out
set -euo pipefail
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source /home/user/summ-mask-benchmark/research/mars2_gates_20260916/slurm/env.sh
source /home/user/summ-mask-benchmark/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MM="${MARS_METRIC_OUT:-/home/user/results/mars_metric}"
SNAP=$(snap "yaxili96/FactCG-DeBERTa-v3-Large"); OUT="$MM/mf_smoke"; mkdir -p "$OUT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PLAN_REPO"
"$PLAN_PY" - <<PY
import json
rows=[json.loads(l) for l in open("$MM/ma/tasks/pairs_ma.jsonl")]
sel=[r for r in rows if r["dataset"]=="unisumeval"][:40]
with open("$OUT/c.jsonl","w") as c, open("$OUT/s.jsonl","w") as s, open("$OUT/r.jsonl","w") as r:
    for p in sel:
        c.write(json.dumps({"text":p["candidate"]})+"\n"); s.write(json.dumps({"text":p["source"]})+"\n"); r.write(json.dumps({"texts":p["references"] or [p["source"]]})+"\n")
print("[mf-smoke] wrote", len(sel), "pairs")
PY
"$PLAN_PY" -m mars score --candidates "$OUT/c.jsonl" --sources "$OUT/s.jsonl" --references "$OUT/r.jsonl" \
  --p-verifier "$SNAP" --r-verifier "$MC/e1/mc-A-lam1-crossenc-inv_seed*/model" --decomposer "seq2seq:/home/user/results/mars3/m32/segmenter" --controls --out "$OUT/scores.jsonl"
"$PLAN_PY" - <<PY
import json, statistics as st
rows=[json.loads(l) for l in open("$OUT/scores.jsonl")]
def m(k): v=[r[k] for r in rows if r.get(k) is not None]; return round(st.mean(v),3) if v else None
print("[mf-smoke] n", len(rows), "P", m("P"), "R", m("R"), "F", m("F"), "facts/cand", m("n_candidate_facts"), "ref facts", m("n_reference_facts"))
print("[mf-smoke] controls", {k: round(st.mean([r["controls"][k] for r in rows if k in r["controls"]]),3) for k in ("P_shuffled_source","R_shuffled_candidate","R_no_candidate")})
print("[mf-smoke] example unsupported:", rows[0]["unsupported"][:2]); print("[mf-smoke] example omitted:", rows[0]["omitted"][:2])
PY
echo "[mf-smoke] done"
