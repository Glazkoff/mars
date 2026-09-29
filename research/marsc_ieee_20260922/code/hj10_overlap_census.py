#!/usr/bin/env python3
"""H-J10a diagnostic (descriptive, no bar): how much of a fact's wording the candidate carries, per pool.

Reads a docs file and b2's cached lexical token-recall family for the same cells
(`lex_token_recall.lex.token_recall.json`, z = 1 - token recall of the clipped fact against the whole
candidate) and reports the distribution of token recall for CONVEYING cells (label 1) and OMITTING cells
(label 0), and the per-triple gap. A pool whose facts restate candidate spans verbatim gives the lexical
counter a near-perfect crossed accuracy by construction; this census says whether that is the case, and by
how much the three pools differ. Nothing is scored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def q(v):
    v = np.array(v, dtype=float)
    return {"n": int(v.size), "mean": float(v.mean()), "p10": float(np.percentile(v, 10)),
            "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "share_ge_0.8": float(np.mean(v >= 0.8)), "share_eq_1": float(np.mean(v >= 0.999))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="append", required=True, help="NAME=<docs.jsonl>:<cache dir>")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = {}
    for spec in a.pool:
        name, rest = spec.split("=", 1)
        docs_f, cache = rest.split(":", 1)
        docs = [json.loads(l) for l in open(docs_f, encoding="utf-8")]
        b = json.loads((Path(cache) / "lex_token_recall.lex.token_recall.json").read_text())
        z = dict(zip(b["cells"], b["z"]))
        conv, omit, gaps = [], [], []
        for c, v in z.items():
            di, si, fi = map(int, c.split("|"))
            lab = docs[di]["summaries"][si]["labels"][fi]
            (conv if lab == 1 else omit).append(1.0 - v)
        for di, d in enumerate(docs):
            for fi in range(len(d["facts"])):
                pos = [1.0 - z[f"{di}|{si}|{fi}"] for si, s in enumerate(d["summaries"])
                       if s["labels"][fi] == 1 and f"{di}|{si}|{fi}" in z]
                neg = [1.0 - z[f"{di}|{si}|{fi}"] for si, s in enumerate(d["summaries"])
                       if s["labels"][fi] == 0 and f"{di}|{si}|{fi}" in z]
                gaps.extend(p - n for p in pos for n in neg)
        rep[name] = {"docs": docs_f, "cells": len(z), "token_recall_conveying_cells": q(conv),
                     "token_recall_omitting_cells": q(omit), "gap_per_triple": q(gaps),
                     "triples_with_positive_gap": float(np.mean(np.array(gaps) > 0)) if gaps else None}
        print(f"[overlap] {name:10s} conveying median {np.median(conv):.3f} (>=0.8: {np.mean(np.array(conv)>=0.8):.2f})"
              f"  omitting median {np.median(omit):.3f}  gap median {np.median(gaps):.3f}  "
              f"triples gap>0 {np.mean(np.array(gaps)>0):.3f}", flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
