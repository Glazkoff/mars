#!/usr/bin/env python3
"""MARS-C R07 -- the crossed inventory x verifier precision design (FULL_REVIEW_2026-09-17 / audit C06).

The paper wrote that the Gemma+ent inventory's recall "is bought with 0.1--0.2 of precision that the humans
locate in its entity units, not in the verifier". That attribution needs a 2x2, and the blinded human sample
covers only three of its four cells: (gemma+ent, human), (distilled, human), (distilled, judge). The fourth,
(gemma+ent, judge), was never annotated, so the inventory main effect and the interaction are not identified
from the human labels alone.

This completes the square with the coverage judge that the SAME human sample validated (Gemma-4-31B-it agrees
with each annotator on 0.90 and on 0.96 of the items they agree on; Appendix "Judge validation"). The judged
omission rate is a proxy, not a human verdict, and is reported as such: the human cells are shown beside it so
a reader can see the offset the proxy carries on the three cells where both exist.

Estimand: among the top-k emitted units of a system, the share the judge answers "omitted" (question `coverage`,
P(B) >= 0.5). Document bootstrap. Then the crossed decomposition at fixed budget:
    inventory effect  = mean over verifiers of (distilled - gemma+ent)
    verifier effect   = mean over inventories of (human - judge)
    interaction       = (d_human - d_judge) difference of the inventory effects
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--judgments", nargs="+", required=True, help="glob(s) of `coverage` judgment files")
    ap.add_argument("--cells", nargs="+", required=True,
                    help="inventory:verifier=system quadruples, e.g. gemma+ent:human=humanfact+div1")
    ap.add_argument("--extra-systems", nargs="*", default=[], help="systems reported but outside the 2x2")
    ap.add_argument("--max-rank", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--human-json", default="", help="R02 human_precision_ranked.json, printed beside the judged cells")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = {t["task_id"]: t for t in load_jsonl(Path(a.tasks))}
    ans = {}
    for g in a.judgments:
        for f in sorted(glob.glob(g)):
            for r in load_jsonl(Path(f)):
                if r.get("question") == "coverage":
                    ans[r["task_id"]] = float(r["p_b"])
    print(f"[r07] {len(tasks)} tasks, {len(ans)} coverage answers", flush=True)
    cells = {}
    for spec in a.cells:
        key, sysname = spec.split("=", 1)
        inv, ver = key.split(":", 1)
        cells[(inv, ver)] = sysname
    systems = sorted(set(cells.values()) | set(a.extra_systems))
    by_doc = defaultdict(lambda: defaultdict(list))                     # system -> pair -> [judged omitted]
    n_missing = 0
    for t in tasks.values():
        for sysname, rank in t["rank"].items():
            if sysname not in systems or rank >= a.max_rank:
                continue
            p = ans.get(t["task_id"])
            if p is None:
                n_missing += 1
                continue
            by_doc[sysname][t["pair_id"]].append(p >= 0.5)
    rng = np.random.default_rng(a.seed)
    pairs_all = sorted(set.intersection(*[set(by_doc[s]) for s in systems])) if systems else []
    print(f"[r07] {len(pairs_all)} pairs carry every system at rank<{a.max_rank}; {n_missing} unjudged (system, unit) slots", flush=True)
    idx = np.arange(len(pairs_all))
    per_pair = {s: np.array([float(np.mean(by_doc[s][p])) for p in pairs_all]) for s in systems}
    boot_idx = [rng.choice(idx, size=len(idx), replace=True) for _ in range(a.n_boot)]
    rep = {"registered": "post-hoc completion of the G6 precision design with the validated coverage judge",
           "max_rank": a.max_rank, "n_pairs": len(pairs_all), "unjudged_slots": n_missing,
           "note": "judge-estimated omission rate among emitted units, NOT a human confirmed-omission rate",
           "systems": {}, "crossed": {}}
    for s in systems:
        d = np.array([per_pair[s][b].mean() for b in boot_idx])
        rep["systems"][s] = {"judged_omission_rate": float(per_pair[s].mean()),
                             "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                             "n_units": int(sum(len(v) for v in by_doc[s].values()))}
        r = rep["systems"][s]
        print(f"[r07] {s:32s} judged-omitted {r['judged_omission_rate']:.4f} "
              f"[{r['ci'][0]:.4f},{r['ci'][1]:.4f}] over {r['n_units']} emitted units", flush=True)
    invs = sorted({k[0] for k in cells}); vers = sorted({k[1] for k in cells})
    if len(invs) == 2 and len(vers) == 2 and len(cells) == 4:
        def eff(fn):
            obs = fn({k: per_pair[v] for k, v in cells.items()})
            d = np.array([fn({k: per_pair[v][b] for k, v in cells.items()}) for b in boot_idx])
            return {"point": float(obs), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        i0, i1 = invs; v0, v1 = vers
        rep["crossed"] = {
            "inventories": invs, "verifiers": vers,
            f"inventory_effect_{i1}_minus_{i0}": eff(lambda c: np.mean([(c[(i1, v)] - c[(i0, v)]).mean() for v in vers])),
            f"verifier_effect_{v1}_minus_{v0}": eff(lambda c: np.mean([(c[(i, v1)] - c[(i, v0)]).mean() for i in invs])),
            "interaction": eff(lambda c: ((c[(i1, v1)] - c[(i0, v1)]).mean() - (c[(i1, v0)] - c[(i0, v0)]).mean()))}
        for k, v in rep["crossed"].items():
            if isinstance(v, dict):
                print(f"[r07 crossed] {k:44s} {v['point']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]", flush=True)
    if a.human_json and Path(a.human_json).exists():
        h = json.load(open(a.human_json))["budgets"].get(str(a.max_rank), {}).get("systems", {})
        rep["human_cells_for_comparison"] = {s: {k: h[s][k] for k in ("both_agree", "n_items_in_budget") if k in h[s]}
                                             for s in h if s in systems}
        for s, v in rep["human_cells_for_comparison"].items():
            ba = v.get("both_agree", {})
            if ba.get("point") is not None:
                print(f"[r07 human] {s:32s} adjudicated human {ba['point']:.4f} "
                      f"[{ba['ci'][0]:.4f},{ba['ci'][1]:.4f}] (n={v.get('n_items_in_budget')})", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "crossed_precision.json").write_text(json.dumps(rep, indent=1))
    print(f"[r07] written {out / 'crossed_precision.json'}", flush=True)


if __name__ == "__main__":
    main()
