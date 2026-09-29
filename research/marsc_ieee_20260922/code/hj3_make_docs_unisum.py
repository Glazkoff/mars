#!/usr/bin/env python3
"""H-J3a -- build the UniSumEval crossed-cell document file in the e0 `docs_<split>.jsonl` schema.

`mc_summary_conditioning.crossed_cells` (imported unmodified by b2_crossed_families.py) reads documents of
the form

    {"doc_key", "resource", "source", "facts": [str, ...],
     "summaries": [{"system", "candidate", "labels": [0/1 per fact], "pair_id", ...}, ...]}

with labels[fi] == 1 meaning the summary CONVEYS fact fi and 0 meaning it OMITS it. UniSumEval
(`g7_unisum/acu_units_unisum.jsonl`, one row per (document, system)) carries exactly that information as
`units` (the document's human-validated key facts, identical across the document's systems) and
`unit_labels` (1 = covered). This script regroups the rows by source document, asserts the fact list is
identical across a document's systems, and writes `docs_unisum.jsonl` plus a census of crossed cells and
triples computed BEFORE any scorer sees a cell. It computes no outcome and loads no model.

Registered in research/marsc_ieee_20260922/PREREG.md (H-J3a).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acu", required=True, help="g7_unisum/acu_units_unisum.jsonl")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in open(a.acu, encoding="utf-8")]
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_doc[hashlib.sha1(r["source"].encode()).hexdigest()[:12]].append(r)

    docs, n_cells, n_triples, docs_with = [], 0, 0, 0
    with open(out / "docs_unisum.jsonl", "w", encoding="utf-8") as fh:
        for key in sorted(by_doc):
            rs = sorted(by_doc[key], key=lambda r: r["pair_id"])
            facts = [u["text"] for u in rs[0]["units"]]
            for r in rs:
                if [u["text"] for u in r["units"]] != facts:
                    raise SystemExit(f"[hj3-docs] ABORT: fact list differs across systems of document {key}")
                if len(r["unit_labels"]) != len(facts):
                    raise SystemExit(f"[hj3-docs] ABORT: {r['pair_id']} has {len(r['unit_labels'])} labels for {len(facts)} facts")
            summaries = [{"system": r["system"], "candidate": r["candidate"],
                          "labels": [1 if x else 0 for x in r["unit_labels"]],
                          "pair_id": r["pair_id"], "split": "unisum", "domain": r["domain"],
                          "words": len(r["candidate"].split())} for r in rs]
            doc = {"doc_key": key, "resource": rs[0]["resource"], "domain": rs[0]["domain"],
                   "source": rs[0]["source"], "splits": ["unisum"], "facts": facts, "summaries": summaries}
            fh.write(json.dumps(doc) + "\n")
            docs.append(doc)
            has = False
            for fi in range(len(facts)):
                pos = [s for s in summaries if s["labels"][fi] == 1]
                neg = [s for s in summaries if s["labels"][fi] == 0]
                if pos and neg:
                    n_triples += len(pos) * len(neg)
                    n_cells += len(pos) + len(neg)
                    has = True
            docs_with += has

    census = {"acu_file": a.acu, "rows": len(rows), "documents": len(docs),
              "documents_with_triples": docs_with, "crossed_cells": n_cells, "same_fact_triples": n_triples,
              "facts_per_document_median": sorted(len(d["facts"]) for d in docs)[len(docs) // 2],
              "label_semantics": "labels[fi]=1 conveys the fact (UniSumEval keyfact_label true), 0 omits it",
              "sha256_docs_unisum": hashlib.sha256((out / "docs_unisum.jsonl").read_bytes()).hexdigest()}
    (out / "census.json").write_text(json.dumps(census, indent=1))
    print(f"[hj3-docs] {json.dumps(census)}", flush=True)


if __name__ == "__main__":
    main()
