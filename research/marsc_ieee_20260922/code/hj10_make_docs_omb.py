#!/usr/bin/env python3
"""H-J10a -- OmissionBench (ComposoAI/OmissionBench, CC BY 4.0, synthetic clinical consultations) in the e0
docs schema that b2_crossed_families.py / mc_summary_conditioning.crossed_cells read.

One document per consultation. Its facts are the statements the benchmark removed from the verified clean
note in that consultation's omission pairs (the pair record's `fact`, de-duplicated by text); its summaries
are the clean twin (every fact present, labels all 1) and each errored note (its own removed fact labelled 0,
every other fact 1, since a pair changes exactly one fact and nothing else). A same-fact crossed triple is
therefore (fact, a note that states it, the note it was removed from); the benchmark's own paired
discrimination is the subset of triples whose conveying note is the clean twin (hj10_twin_metric.py).

Two docs files: `docs_omb_complete.jsonl` (residual level `complete`: the fact is truly absent from the
errored note; the registered primary set) and `docs_omb_all.jsonl` (every evaluation omission pair, the
partial residual levels included; secondary). Nothing is scored here; the census is written before any
scorer runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

CLIP = 380  # the deployed verifier's candidate clip, in words


def first_diff_word(a: str, b: str) -> int:
    """Index of the first word at which the errored note departs from its clean twin."""
    wa, wb = a.split(), b.split()
    i = 0
    while i < len(wa) and i < len(wb) and wa[i] == wb[i]:
        i += 1
    return i


def build(pairs: list[dict], transcripts: Path, keep) -> tuple[list[dict], dict]:
    by_key: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        if p["eval_set"] and p["type"] == "omit" and keep(p):
            by_key[p["key"]].append(p)
    docs, n_cells, n_triples, beyond_clip, n_pairs = [], 0, 0, 0, 0
    for key in sorted(by_key):
        ps = sorted(by_key[key], key=lambda p: p["pair_id"])
        clean = {p["clean"] for p in ps}
        if len(clean) != 1:
            raise SystemExit(f"[hj10-docs] ABORT: {key} has {len(clean)} distinct clean twins")
        clean = ps[0]["clean"]
        facts: list[str] = []
        for p in ps:
            if p["fact"] not in facts:
                facts.append(p["fact"])
        stratum, cid = ps[0]["stratum"], ps[0]["id"]
        src = transcripts / stratum / f"{cid}.txt"
        if not src.exists():
            raise SystemExit(f"[hj10-docs] ABORT: transcript missing for {key}: {src}")
        summaries = [{"system": "clean", "candidate": clean, "labels": [1] * len(facts), "pair_id": key,
                      "split": "omb", "domain": stratum, "words": len(clean.split())}]
        for p in ps:
            fi = facts.index(p["fact"])
            labels = [1] * len(facts)
            labels[fi] = 0
            d0 = first_diff_word(clean, p["errored"])
            beyond_clip += d0 >= CLIP
            n_pairs += 1
            summaries.append({"system": "errored", "candidate": p["errored"], "labels": labels,
                              "pair_id": p["pair_id"], "split": "omb", "domain": stratum,
                              "words": len(p["errored"].split()), "fact_index": fi, "class": p["class"],
                              "residual_level": p["residual_level"], "severity": p["severity"],
                              "cell": p.get("cell"), "source_cohort": p["source"],
                              "first_diff_word": d0})
        doc = {"doc_key": key, "resource": "omissionbench", "domain": stratum, "consultation": cid,
               "source": src.read_text(encoding="utf-8"), "splits": ["omb"], "facts": facts,
               "summaries": summaries}
        docs.append(doc)
        for fi in range(len(facts)):
            pos = [s for s in summaries if s["labels"][fi] == 1]
            neg = [s for s in summaries if s["labels"][fi] == 0]
            if pos and neg:
                n_triples += len(pos) * len(neg)
                n_cells += len(pos) + len(neg)
    err = [s for d in docs for s in d["summaries"] if s["system"] == "errored"]
    words = [s["words"] for d in docs for s in d["summaries"]]
    census = {"consultations": len(docs), "omission_pairs": n_pairs, "facts": sum(len(d["facts"]) for d in docs),
              "crossed_cells": n_cells, "same_fact_triples": n_triples,
              "pairs_by_stratum": dict(Counter(s["domain"] for s in err)),
              "pairs_by_residual_level": dict(Counter(str(s["residual_level"]) for s in err)),
              "pairs_by_severity": dict(Counter(s["severity"] for s in err)),
              "pairs_by_cell": dict(Counter(str(s["cell"]) for s in err)),
              "note_words_median": st.median(words), "note_words_max": max(words),
              "notes_over_clip_share": sum(w > CLIP for w in words) / len(words),
              "pairs_first_removed_site_beyond_clip": beyond_clip,
              "pairs_first_removed_site_beyond_clip_share": beyond_clip / n_pairs,
              "clip_words": CLIP,
              "label_semantics": "labels[fi]=1 the note states fact fi (clean twin, or an errored note of another "
                                 "fact); 0 the note is the one fact fi was removed from"}
    return docs, census


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="pairs/dataset_v2.json")
    ap.add_argument("--transcripts", required=True, help="transcripts/ directory (stratum/id.txt)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    blob = json.load(open(a.pairs, encoding="utf-8"))
    pairs = blob["pairs"]
    report = {"pairs_file": a.pairs, "dataset_version": blob.get("dataset_version"),
              "records": len(pairs), "evaluation_omission_pairs": sum(1 for p in pairs if p["eval_set"] and p["type"] == "omit")}
    for name, keep in (("complete", lambda p: p["residual_level"] == "complete"), ("all", lambda p: True)):
        docs, census = build(pairs, Path(a.transcripts), keep)
        f = out / f"docs_omb_{name}.jsonl"
        with open(f, "w", encoding="utf-8") as fh:
            for d in docs:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        census["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
        report[name] = census
        print(f"[hj10-docs] {name}: {json.dumps({k: v for k, v in census.items() if not isinstance(v, dict)})}", flush=True)
    (out / "census_omb.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
