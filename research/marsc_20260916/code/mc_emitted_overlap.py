#!/usr/bin/env python3
"""MARS-C R02b -- how much of the annotated emission survives the estimator repair?

The blinded human sample was drawn from the emissions of the mis-aggregated validation consensus. After the
repair the same systems emit a partly different top ten, so the honest questions are (a) how much of the
annotated set the corrected emitter still emits, and (b) whether the precision estimate computed on the
annotated items that the corrected emitter still ranks in its top ten agrees with the one computed on the
annotated set as a whole. This script answers (a); mc_human_rank.py run against the corrected dump answers (b).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def topk(rows, k):
    out = defaultdict(dict)
    for r in rows:
        if not r.get("emitted"):
            continue
        out[r["system"]][r["pair_id"]] = [e["text"] for e in r["emitted"][:k]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-annotated", required=True)
    ap.add_argument("--corrected", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, B = topk(load(a.as_annotated), a.k), topk(load(a.corrected), a.k)
    rep = {"k": a.k, "systems": {}}
    for s in sorted(set(A) & set(B)):
        pairs = sorted(set(A[s]) & set(B[s]))
        inter = tot_a = tot_b = 0
        for p in pairs:
            sa, sb = set(A[s][p]), set(B[s][p])
            inter += len(sa & sb); tot_a += len(sa); tot_b += len(sb)
        rep["systems"][s] = {"n_pairs": len(pairs), "units_as_annotated": tot_a, "units_corrected": tot_b,
                             "shared": inter, "jaccard_like_recall": inter / max(1, tot_a)}
        r = rep["systems"][s]
        print("[r02b] %-30s top-%d overlap %.4f (%d of %d annotated-emitter units still emitted)"
              % (s, a.k, r["jaccard_like_recall"], inter, tot_a), flush=True)
    key = {r["item_id"]: r for r in load(a.key)}
    still = defaultdict(lambda: [0, 0])
    for it in key.values():
        s, p = it["system"], it["pair_id"]
        if s in B and p in B[s]:
            still[s][1] += 1
            if it["fact"] in B[s][p]:
                still[s][0] += 1
        else:
            still[s][1] += 1
    rep["annotated_items_still_in_corrected_topk"] = {s: {"in": v[0], "of": v[1], "share": v[0] / max(1, v[1])}
                                                     for s, v in still.items()}
    for s, v in rep["annotated_items_still_in_corrected_topk"].items():
        print("[r02b] %-30s %d of %d ANNOTATED items are in the corrected top-%d (%.3f)"
              % (s, v["in"], v["of"], a.k, v["share"]), flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "emitted_overlap.json").write_text(json.dumps(rep, indent=1))
    print("[r02b] written", out / "emitted_overlap.json", flush=True)


if __name__ == "__main__":
    main()
