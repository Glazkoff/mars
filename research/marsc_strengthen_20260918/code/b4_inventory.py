#!/usr/bin/env python3
"""B4 -- build the unseen extractor's unit inventory under the frozen `gemma+ent` recipe.

The recipe is copied, line for line in effect, from research/mars3_20260916/code/m32_inventory.py's
`gemma+ent` branch: the inventory of a source document is its LLM propositions (span = the sentence they came
from) followed by the spaCy entity units of that document, cached per document and reused across every pair
that shares it. m32_inventory.py hard-codes `--variant` to four names and branches on them, so a third
extractor cannot be expressed there; that file is shared with live sessions and is NOT modified. Only the
source of the propositions changes here -- everything downstream (span, dict shape, order, cache) is identical.

Also recomputes, on exactly the m32 definitions, the human-omitted-ACU ceiling of the new inventory (weak:
same sentence + >= 1 shared content token; strict: the unit covers >= 60 % of the ACU's content tokens), so
the unseen extractor's inventory quality can be read beside the development extractors' published ceilings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for p in (str(REPO), str(REPO / "research" / "mars2_gates_20260916" / "code")):
    if p not in sys.path:
        sys.path.insert(0, p)
import common as gates  # noqa: E402


def props_by_doc(files: list[str]) -> dict[str, list[dict]]:
    """m32_inventory.props_by_doc, unchanged."""
    out = defaultdict(list)
    for f in files:
        for r in gates.load_jsonl(Path(f)):
            for p in r.get("props", []):
                out[r["doc_key"]].append({"kind": "prop_llm", "text": p, "start": r["start"], "end": r["end"],
                                          "label": "", "args": {}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="B1 pairs_<split>.jsonl (carries acu_units for the ceiling)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--variant", default="qwenprops+ent", help="inventory name; the recipe is gemma+ent's")
    ap.add_argument("--props", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-doc-coverage", type=float, default=0.98,
                    help="abort if fewer than this share of source documents carry at least one proposition")
    ap.add_argument("--restrict-to-acu", default="",
                    help="an ACU-labelled units file; keep only pairs whose SOURCE DOCUMENT appears in it. "
                         "B4 is confirmed against the sealed TEST ACU labels, so the inventory is built over "
                         "exactly the documents that evaluation scores. b1/pairs_test.jsonl additionally "
                         "carries natural770, a B1 TRAINING resource that no unseen-extractor confirmation may "
                         "touch and that acu_units_test.jsonl does not label.")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pairs = gates.load_jsonl(Path(a.pairs))
    if a.restrict_to_acu:
        keep = {hashlib.sha1(r["source"].encode()).hexdigest()[:12] for r in gates.load_jsonl(Path(a.restrict_to_acu))}
        before = len(pairs)
        pairs = [r for r in pairs if hashlib.sha1(r["source"].encode()).hexdigest()[:12] in keep]
        print(f"[b4-inventory] --restrict-to-acu {a.restrict_to_acu}: {len(keep)} labelled documents; "
              f"pairs {before} -> {len(pairs)}", flush=True)
        assert pairs, "--restrict-to-acu removed every pair; check that the ACU file matches the split"
    if a.limit:
        pairs = pairs[:a.limit]
    from mars_v2.units import get_extractor
    extractor = get_extractor("entity")
    props = props_by_doc(a.props)
    print(f"[b4-inventory] {len(pairs)} pairs; propositions for {len(props)} documents from {len(a.props)} shard file(s)", flush=True)

    want_docs = {hashlib.sha1(r["source"].encode()).hexdigest()[:12] for r in pairs}
    have = len(want_docs & set(props))
    cov0 = have / max(1, len(want_docs))
    print(f"[b4-inventory] pre-check: {have}/{len(want_docs)} source documents carry at least one proposition ({cov0:.4f})", flush=True)
    if cov0 < a.min_doc_coverage:
        raise SystemExit(f"[b4-inventory] ABORT before writing: document coverage {cov0:.4f} < {a.min_doc_coverage}; "
                         f"the decomposition shards are incomplete")

    inv_cache: dict[str, list[dict]] = {}

    def inventory(src: str) -> list[dict]:
        k = hashlib.sha1(src.encode()).hexdigest()[:12]
        if k in inv_cache:
            return inv_cache[k]
        units = list(props.get(k, []))
        units += [u.to_dict() for u in extractor(src)]
        inv_cache[k] = units
        return units

    stat = defaultdict(Counter); n_units = []; n_props = []
    units_path = out / f"units_{a.variant}_{a.split}.jsonl"
    with open(units_path, "w", encoding="utf-8") as fh:
        for r in pairs:
            units = inventory(r["source"]); n_units.append(len(units))
            n_props.append(sum(1 for u in units if u["kind"] == "prop_llm"))
            fh.write(json.dumps({"pair_id": r["pair_id"], "resource": r["resource"], "system": r["system"],
                                 "split": a.split, "source": r["source"], "candidate": r["candidate"],
                                 "units": units}) + "\n")
            for acu in r.get("acu_units") or []:
                s, e = gates.best_sentence_span(r["source"], acu["text"])
                ct = set(gates.tokens(acu["text"], content_only=True)) or set(gates.tokens(acu["text"]))
                same_sent = [u for u in units if u["start"] < e and u["end"] > s]
                weak = [u for u in same_sent if ct & set(gates.tokens(u["text"]))]
                strict = [u for u in same_sent if len(ct & set(gates.tokens(u["text"]))) >= 0.6 * len(ct)]
                res = r["resource"]; om = acu["label"] == 0
                stat[res]["acus"] += 1; stat[res]["weak"] += bool(weak); stat[res]["strict"] += bool(strict)
                if om:
                    stat[res]["omitted"] += 1; stat[res]["omitted_weak"] += bool(weak); stat[res]["omitted_strict"] += bool(strict)

    docs_total = len(inv_cache)
    docs_with_props = sum(1 for k in inv_cache if props.get(k))
    cov = docs_with_props / max(1, docs_total)
    cells = {res: {"human_omitted": c["omitted"], "ceiling_weak": c["omitted_weak"] / max(1, c["omitted"]),
                   "ceiling_strict": c["omitted_strict"] / max(1, c["omitted"]),
                   "all_acus_weak": c["weak"] / max(1, c["acus"])} for res, c in stat.items()}
    tot_om = sum(c["omitted"] for c in stat.values())
    rep = {"variant": a.variant, "split": a.split, "recipe": "gemma+ent (m32_inventory.py), propositions from the unseen extractor",
           "pairs": len(pairs), "documents": docs_total, "documents_with_propositions": docs_with_props,
           "document_coverage": cov, "units_total": sum(n_units),
           "mean_units_per_pair": sum(n_units) / max(1, len(n_units)),
           "mean_propositions_per_pair": sum(n_props) / max(1, len(n_props)),
           "ceiling_weak_pooled": sum(c["omitted_weak"] for c in stat.values()) / max(1, tot_om),
           "ceiling_strict_pooled": sum(c["omitted_strict"] for c in stat.values()) / max(1, tot_om),
           "cells": cells, "units_file": str(units_path)}
    (out / f"inventory_{a.variant}_{a.split}.json").write_text(json.dumps(rep, indent=1))
    print(f"[b4-inventory] {a.variant}/{a.split}: {rep['mean_units_per_pair']:.1f} units/pair "
          f"({rep['mean_propositions_per_pair']:.1f} LLM props); document coverage {cov:.4f}; "
          f"human-omitted ACU ceiling weak {rep['ceiling_weak_pooled']:.3f} strict {rep['ceiling_strict_pooled']:.3f}", flush=True)
    for k, v in cells.items():
        print(f"[b4-inventory] {k}: ceiling weak {v['ceiling_weak']:.3f} strict {v['ceiling_strict']:.3f} (n_omitted {v['human_omitted']})", flush=True)
    if cov < a.min_doc_coverage:
        raise SystemExit(f"[b4-inventory] ABORT: only {docs_with_props}/{docs_total} documents carry a proposition "
                         f"({cov:.4f} < {a.min_doc_coverage}); the decomposition shards are incomplete")
    print("[b4-inventory] done", flush=True)


if __name__ == "__main__":
    main()
