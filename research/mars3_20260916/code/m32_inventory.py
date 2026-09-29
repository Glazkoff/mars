#!/usr/bin/env python3
"""M32 -- build unit inventories for every pair of a split and measure the human-ACU ceiling of each.

Variants: spacy (B1 extractor, cap 40), gemma (LLM propositions; span = their sentence), gemma+ent (propositions + spaCy
entities), distilled (propositions from the distilled segmenter). Ceiling = share of human-omitted ACUs with an aligned
unit (D1 rule: same sentence + >= 1 shared content token; strict rule: unit covers >= 60 % of the ACU's content tokens).
Writes <out>/units_<variant>_<split>.jsonl (pairs with units) and appends to <out>/ceilings.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import m3common as m3
from m3common import gates

MAX_SPACY = 40


def props_by_doc(files: list[str]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for f in files:
        for r in gates.load_jsonl(Path(f)):
            for p in r.get("props", []):
                out[r["doc_key"]].append({"kind": "prop_llm", "text": p, "start": r["start"], "end": r["end"], "label": "", "args": {}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--variant", choices=["spacy", "gemma", "gemma+ent", "distilled"], required=True)
    ap.add_argument("--props", nargs="*", default=[], help="decompose-format files (gemma / distilled)")
    ap.add_argument("--out", default=str(m3.M3_OUT / "m32"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pairs = gates.load_jsonl(gates.B1_DIR / f"pairs_{a.split}.jsonl")
    if a.limit:
        pairs = pairs[:a.limit]
    need_spacy = a.variant in ("spacy", "gemma+ent")
    extractor = None
    if need_spacy:
        from mars_v2.units import get_extractor
        extractor = get_extractor("entity+proposition" if a.variant == "spacy" else "entity")
    props = props_by_doc(a.props) if a.variant != "spacy" else {}
    inv_cache = {}

    def inventory(src: str, row: dict) -> list[dict]:
        k = hashlib.sha1(src.encode()).hexdigest()[:12]
        if k in inv_cache:
            return inv_cache[k]
        units = []
        if a.variant == "spacy":
            units = row.get("units") or [u.to_dict() for u in extractor(src)[:MAX_SPACY]]
        else:
            units = list(props.get(k, []))
            if a.variant == "gemma+ent":
                units += [u.to_dict() for u in extractor(src)]
        inv_cache[k] = units
        return units

    stat = defaultdict(Counter); n_units = []
    with open(out / f"units_{a.variant}_{a.split}.jsonl", "w", encoding="utf-8") as fh:
        for r in pairs:
            units = inventory(r["source"], r); n_units.append(len(units))
            fh.write(json.dumps({"pair_id": r["pair_id"], "resource": r["resource"], "system": r["system"], "split": a.split,
                                 "source": r["source"], "candidate": r["candidate"], "units": units}) + "\n")
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
    cells = {res: {"human_omitted": c["omitted"], "ceiling_weak": c["omitted_weak"] / max(1, c["omitted"]),
                   "ceiling_strict": c["omitted_strict"] / max(1, c["omitted"]), "all_acus_weak": c["weak"] / max(1, c["acus"])} for res, c in stat.items()}
    tot_om = sum(c["omitted"] for c in stat.values())
    rep = {"variant": a.variant, "split": a.split, "pairs": len(pairs), "mean_units_per_pair": sum(n_units) / max(1, len(n_units)),
           "ceiling_weak_pooled": sum(c["omitted_weak"] for c in stat.values()) / max(1, tot_om),
           "ceiling_strict_pooled": sum(c["omitted_strict"] for c in stat.values()) / max(1, tot_om), "cells": cells}
    cp = out / "ceilings.json"; allrep = json.load(open(cp)) if cp.exists() else {}
    allrep[f"{a.variant}/{a.split}"] = rep; json.dump(allrep, open(cp, "w"), indent=1)
    print(f"[m32-inventory] {a.variant}/{a.split}: {rep['mean_units_per_pair']:.1f} units/pair; human-omitted ACU ceiling weak {rep['ceiling_weak_pooled']:.3f} strict {rep['ceiling_strict_pooled']:.3f}; " +
          ", ".join(f"{k} {v['ceiling_weak']:.2f}/{v['ceiling_strict']:.2f}" for k, v in cells.items()), flush=True)


if __name__ == "__main__":
    main()
