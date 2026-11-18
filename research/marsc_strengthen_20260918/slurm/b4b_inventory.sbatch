#!/bin/bash
# MARS-C strengthening B4b -- the unseen extractor's unit inventory under the frozen gemma+ent recipe.
#
# Consolidates the two decomposition shards, asserts that both shards froze the SAME model revision, prompt
# sha256 and decoding contract (an integrity check on the pre-registration itself), copies one of them to
# freeze.json, and then builds units_qwenprops+ent_test.jsonl: the unseen extractor's propositions followed by
# the same spaCy entity units the development inventory uses, cached per source document exactly as
# m32_inventory.py's gemma+ent branch does. m32_inventory.py hard-codes its four variant names and is shared
# with live sessions, so the recipe is re-implemented in this campaign's own code rather than edited there.
#
# Also prints the new inventory's human-omitted-ACU ceiling on the m32 definitions, so the unseen extractor's
# inventory quality can be read beside the development extractors' published ceilings before any verifier runs.
#SBATCH --job-name=mars-b4b-inventory
#SBATCH --partition=infer
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/user/results/marsc_strengthen/logs/%x-%A.out
[ "${MC_TEST_REREAD:-}" = "yes" ] || { echo "[b4b] REFUSED: MC_TEST_REREAD=yes not set (sealed TEST split)"; exit 3; }
source /home/user/summ-mask-benchmark/slurm/cluster-B/plan2026/common.sh
source $PLAN_REPO/research/mars2_gates_20260916/slurm/env.sh
source $PLAN_REPO/research/mars3_20260916/slurm/env.sh
export MC="${MARSC_OUT:-/home/user/results/marsc}"; MCCODE="$PLAN_REPO/research/marsc_20260916/code"
SCODE="$PLAN_REPO/research/marsc_strengthen_20260918/code"
OUT="${B4_OUT:-/home/user/results/marsc_strengthen/b4}"; mkdir -p "$OUT/inv"

echo "===== B4b.0 the freeze manifests agree across shards"
"$PLAN_PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); fr = sorted((out / "freeze").glob("freeze_shard*.json"))
assert len(fr) == 2, f"expected 2 shard freeze manifests, found {[f.name for f in fr]}"
objs = [json.loads(f.read_text()) for f in fr]
keys = ("model", "prompt", "parser")
for k in keys:
    ref = objs[0][k]
    for f, o in zip(fr[1:], objs[1:]):
        assert o[k] == ref, f"freeze manifests disagree on {k!r}: {fr[0].name} vs {f.name}"
dec = [{kk: vv for kk, vv in o["decoding"].items() if kk != "device"} for o in objs]
assert dec[0] == dec[1], "freeze manifests disagree on the decoding contract"
tasks = [o["tasks"] for o in objs]
assert tasks[0]["sha256"] == tasks[1]["sha256"], "the two shards read different task files"
merged = dict(objs[0]); merged["shards"] = [o["tasks"]["shard"] for o in objs]
merged["tasks"] = {kk: vv for kk, vv in tasks[0].items() if kk != "shard"}
merged["guards"] = objs[0]["guards"]
(out / "freeze" / "freeze.json").write_text(json.dumps(merged, indent=1))
print(f"[b4b] freeze agreed across {len(fr)} shards; revision {merged['model']['resolved_revision']} "
      f"prompt sha256 {merged['prompt']['sha256']}", flush=True)
PY

echo "===== B4b.1 shard statistics (blank-decode and empty-parse rates of the unseen extractor)"
cat "$OUT"/props/props_qwen_test_shard*.jsonl.stats.json

echo "===== B4b.2 the inventory"
"$PLAN_PY" "$SCODE/b4_inventory.py" \
  --pairs "$B1_DIR/pairs_test.jsonl" --restrict-to-acu "$G/b18/acu_units_test.jsonl" --split test --variant "qwenprops+ent" \
  --props "$OUT"/props/props_qwen_test_shard0.jsonl "$OUT"/props/props_qwen_test_shard1.jsonl \
  --out "$OUT/inv"

echo "===== B4b.3 the development inventory, for the record (not recomputed, only reported)"
"$PLAN_PY" - <<'PY'
import json, os
p = "/home/user/results/mars3/m32/ceilings.json"
c = json.load(open(p))
for k in ("gemma+ent/test", "distilled/test"):
    if k in c:
        r = c[k]
        print(f"[b4b] development {k}: {r['mean_units_per_pair']:.1f} units/pair "
              f"ceiling weak {r['ceiling_weak_pooled']:.3f} strict {r['ceiling_strict_pooled']:.3f}", flush=True)
PY
echo "[b4b] done"
